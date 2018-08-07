import torch
import torch.utils.data as data
from torch.utils.data.dataset import random_split


class Solver:
    def __init__(self, model, optim, loss_fn, metric_fn):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

    def fit(self, dataset, n_epochs, batch_size=32, train_split=0.8, print_cnt=0):
        n_train = int(len(dataset) * train_split)
        n_val = len(dataset) - n_train
        dataset_train, dataset_val = random_split(dataset, [n_train, n_val])
        print("total:{}, n_train:{}, n_val:{}".format(len(dataset), len(dataset_train), len(dataset_val)))
        loader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        loader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        for i in range(1, n_epochs + 1):
            total_loss = 0
            cnt = 0
            n = 0
            for x, y in loader_train:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                cnt += batch_size
                n += batch_size
                if 0 < print_cnt < cnt:
                    cnt -= print_cnt
                    print("epoch:{} {}, loss:{}".format(i, n / n_train, loss.item()))

            total_metric = 0
            with torch.no_grad():
                for x, y in loader_val:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_pred = self.model(x)
                    total_metric += self.metric_fn(y_pred, y)

            print("epoch:{} loss:{} metric:{}".format(i, total_loss / n_train, total_metric / n_val))
