import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.hidden_size = in_ch * kernel_size * kernel_size

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = self.out_ch
        self.kernel_size = kernel_size
        self.rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)

    def forward(self, x, state):
        if state is None:
            w = x.new_zeros((1, self.hidden_size))
            state = x.new_zeros((1, self.hidden_size))
            state = state, state
        else:
            w, state = state
        state = self.rnn(w, state)
        w, _ = state

        w = w.view(self.in_ch, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, w, padding=1, groups=self.groups)
        w = w.view(1, -1)
        return x, (w, state)


class RNNUpConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = UpConv2d(in_ch, out_ch)
        self.lstm = RNNConv2d(out_ch, out_ch, 3)

    def forward(self, inputs):
        xs, ss = inputs
        xs = self.conv(xs)
        x, s = self.lstm(xs.pop(), ss.pop(0))
        xs.append(x)
        ss.append(s)
        return xs, ss


class LSTMConv2d(nn.Module):

    def __init__(self, in_ch, h_ch):
        super().__init__()

        self.in_ch = in_ch
        self.h_ch = h_ch

        self.conv = nn.Conv2d(self.in_ch + self.h_ch, 4 * self.h_ch, 1)

    def forward(self, x, state):
        if state is None:
            batch_size, _, height, weight = x.shape
            state = x.new_zeros((batch_size, self.h_ch, height, weight))
            state = (state, state)

        hidden, cell = state

        combined = torch.cat([x, hidden], dim=1)

        i, f, g, o = torch.split(self.conv(combined), self.h_ch, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        return hidden, (hidden, cell)


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
        xs.append(self.down(xs[-1]))
        return xs


class UpConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SimpleConv2d(in_ch + out_ch, out_ch)

    def forward(self, xs):
        x1 = xs.pop()
        x2 = xs.pop()  # compare to xs[-1], this looks better
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        xs.append(x)
        return xs


class LSTMDownConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DownConv2d(in_ch, out_ch)
        self.lstm = LSTMConv2d(out_ch, out_ch)

    def forward(self, inputs):
        xs, ss = inputs
        xs = self.conv(xs)
        x, s = self.lstm(xs.pop(), ss.pop(0))
        xs.append(x)
        ss.append(s)
        return xs, ss


class LSTMUpConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = UpConv2d(in_ch, out_ch)
        self.lstm = LSTMConv2d(out_ch, out_ch)

    def forward(self, inputs):
        xs, ss = inputs
        xs = self.conv(xs)
        x, s = self.lstm(xs.pop(), ss.pop(0))
        xs.append(x)
        ss.append(s)
        return xs, ss


class BaselineModel(nn.Module):

    def __init__(self, in_ch, out_ch, n_seq, batch_first=False):
        super().__init__()

        self.n_seq = n_seq
        self.batch_first = batch_first
        self.num_layers = 3

        self.in_conv = SimpleConv2d(in_ch, 16)
        self.down = nn.Sequential(
            LSTMDownConv2d(16, 32),
            LSTMDownConv2d(32, 64),
            LSTMDownConv2d(64, 64),
        )
        self.up = nn.Sequential(
            RNNUpConv2d(64, 64),
            RNNUpConv2d(64, 32),
            RNNUpConv2d(32, 16),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs, state=None):
        if self.batch_first:
            xs = xs.transpose(0, 1)
        if state is None:
            down_state = [None for _ in range(self.num_layers)]
            up_state = [None for _ in range(self.num_layers)]
        else:
            down_state, up_state = state

        out = []
        for x in xs:
            x = self.in_conv(x)
            x = [x], down_state
            x, down_state = self.down(x)
            # x = self.down([x])
            # x = self.up(x)
            x = x, up_state
            x, up_state = self.up(x)
            x = x[0]
            x = self.out_conv(x)
            out.append(x)
        for _ in range(self.n_seq):
            x = self.in_conv(x)
            x = [x], down_state
            x, down_state = self.down(x)
            # x = self.down([x])
            # self.up(x)
            x = x, up_state
            x, up_state = self.up(x)
            x = x[0]
            x = self.out_conv(x)
            out.append(x)
        out = out[-self.n_seq:]
        out = torch.stack(out)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)
