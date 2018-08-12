import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv2d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class DownConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            SimpleConv2d(in_ch, out_ch)
        )

    def forward(self, xs):
        xs, ss = xs
        xs.append(self.down(xs[-1]))
        return xs, ss


class UpConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SimpleConv2d(in_ch + out_ch, out_ch)

    def forward(self, xs):
        xs, ss = xs
        x1 = xs.pop()
        x2 = xs.pop()  # compare to xs[-1], this looks better
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        xs.append(x)
        return xs, ss


class LSTMConv2dCell(nn.Module):

    def __init__(self, in_ch, h_ch, kernel_size, padding=0, groups=1):
        super().__init__()

        self.in_ch = in_ch
        self.h_ch = h_ch

        self.conv = nn.Conv2d(self.in_ch + self.h_ch, 4 * self.h_ch,
                              kernel_size=kernel_size, padding=padding, groups=groups)

    def forward(self, x, state):
        if state is None:
            batch_size, _, height, weight = x.shape
            state = x.new_zeros((batch_size, self.h_ch, height, weight))
            state = (state, state)

        hidden, cell = state

        combined = torch.cat([x, hidden], dim=1)
        combined = self.conv(combined)

        i, f, g, o = torch.split(combined, self.h_ch, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        return hidden, (hidden, cell)


class TransConv2dCell(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.hidden_size = out_ch * kernel_size * kernel_size

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)

    def forward(self, x, state):
        if state is None:
            w = x.new_ones((1, self.hidden_size))
            state = x.new_zeros((1, self.hidden_size))
            state = state, state
        else:
            w, state = state
        state = self.rnn(w, state)
        w, _ = state

        w = w.view(self.out_ch, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, w, padding=1, groups=self.in_ch)
        w = w.view(1, -1)
        return x, (w, state)


class UVAlignCell(nn.Module):

    def __init__(self, n_ch, h_ch=8):
        super().__init__()

        self.rnn = LSTMConv2dCell(n_ch, h_ch, 1)
        self.conv = nn.Conv2d(h_ch, 2, 7)

        self.u_align = nn.AdaptiveAvgPool2d((1, 3))
        self.v_align = nn.AdaptiveAvgPool2d((3, 1))

    def forward(self, x, state):
        hidden, state = self.rnn(x, state)
        uv = F.sigmoid(self.conv(hidden))

        batch_size, n_ch, h, w = x.shape
        u_align = self.u_align(uv[:, :1]).expand((-1, n_ch, -1, -1))
        v_align = self.v_align(uv[:, 1:]).expand((-1, n_ch, -1, -1))
        u_align = u_align.contiguous().view(batch_size * n_ch, 1, 1, 3)
        v_align = v_align.contiguous().view(batch_size * n_ch, 1, 3, 1)
        x = x.view(1, batch_size * n_ch, h, w)
        x = F.conv2d(x, u_align, padding=(0, 1), groups=batch_size * n_ch)
        x = F.conv2d(x, v_align, padding=(1, 0), groups=batch_size * n_ch)
        x = x.view(batch_size, n_ch, h, w)

        return x, state


class RNNConv2dBase(nn.Module):

    def __init__(self, mode, in_ch, h_ch):
        super().__init__()
        if mode == "LSTM":
            self.rnn = LSTMConv2dCell(in_ch, h_ch, 1)
        elif mode == "Trans":
            self.rnn = TransConv2dCell(in_ch, h_ch, 3)
        elif mode == "UVAlign":
            self.rnn = UVAlignCell(in_ch)

    def forward(self, inputs):
        xs, ss = inputs
        x, s = xs.pop(), ss.pop()
        x, s = self.rnn(x, s)
        xs.append(x)
        ss.insert(0, s)
        return xs, ss


class LSTMConv2d(RNNConv2dBase):

    def __init__(self, in_ch, h_ch):
        super().__init__("LSTM", in_ch, h_ch)


class TransConv2d(RNNConv2dBase):

    def __init__(self, in_ch, h_ch):
        super().__init__("Trans", in_ch, h_ch)


class UVAlign(RNNConv2dBase):

    def __init__(self, in_ch, h_ch):
        super().__init__("UVAlign", in_ch, h_ch)


class RainfallLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.intervals = [0, 0.05, 0.10, 0.15, 0.20, 0.40, 1.01]
        self.weights = [1, 2, 5, 10, 20, 1]
        # self.weights = [20, 10, 5, 5, 1, 1]

    def forward(self, y_pred, y_true):
        mask = [y_true < interval for interval in self.intervals]
        weight = [w * (u - l).to(torch.float32) for w, l, u in zip(self.weights, mask[:-1], mask[1:])]
        weight = sum(weight)

        loss = weight * (y_pred - y_true) ** 2
        return torch.mean(loss)


class BaselineModel(nn.Module):

    def __init__(self, in_ch, out_ch, n_seq, batch_first=False):
        super().__init__()

        self.n_seq = n_seq
        self.batch_first = batch_first
        self.down_layers = 3
        self.up_layers = 3

        self.in_conv = SimpleConv2d(in_ch, 48)
        self.down = nn.Sequential(
            # LSTMConv2d(48, 48),
            DownConv2d(48, 64),
            # LSTMConv2d(64, 64),
            DownConv2d(64, 96),
            # LSTMConv2d(96, 96),
        )
        self.up = nn.Sequential(
            LSTMConv2d(96, 96),
            UpConv2d(96, 64),
            LSTMConv2d(64, 64),
            UpConv2d(64, 48),
            LSTMConv2d(48, 48),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(48, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs, state=None):
        if self.batch_first:
            xs = xs.transpose(0, 1)
        if state is None:
            down_state = [None for _ in range(self.down_layers)]
            up_state = [None for _ in range(self.up_layers)]
            state = down_state, up_state

        out = []
        for x in xs:
            x, state = self.predict(x, state)
            out.append(x)
        for _ in range(self.n_seq - 1):
            x, state = self.predict(x, state)
            out.append(x)
        out = out[-self.n_seq:]
        out = torch.stack(out)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out

    def predict(self, x, state):
        down_state, up_state = state
        x = self.in_conv(x)
        x = [x], down_state
        x, down_state = self.down(x)
        x = x, up_state
        x, up_state = self.up(x)
        x = x[0]
        x = self.out_conv(x)
        return x, (down_state, up_state)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)
