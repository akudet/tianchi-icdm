import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class RandomSample:
    def __init__(self, n_samples, step):
        self.n_samples = n_samples
        self.step = step

    def __call__(self, samples):
        import random
        n = self.n_samples * self.step
        s = random.randrange(0, len(samples) - n)
        return samples[s: s + n: self.step]


class BatchLoad:
    def __init__(self):
        from torchvision.datasets.folder import default_loader
        self.loader = default_loader

    def __call__(self, img_files):
        return [self.loader(img_file) for img_file in img_files]


class Split:
    def __init__(self, n_seq):
        self.n_seq = n_seq

    def __call__(self, x):
        return x[:-self.n_seq], x[-self.n_seq:]


class BatchApply:

    def __init__(self, trans):
        self.trans = trans

    def __call__(self, xs):
        return [self.trans(x) for x in xs]


class BatchToTensor:

    def __init__(self):
        self.trans = transforms.ToTensor()

    def __call__(self, imgs):
        return torch.stack([self.trans(img) for img in imgs])


def get_dataset(root, is_train=True):
    if is_train:
        transform_train = transforms.Compose([
            RandomSample(12, 5),
            BatchLoad(),
            BatchApply(transforms.Resize(256)),
            BatchToTensor(),
            Split(6),
        ])
        dataset = SRADDataset(root)
        dataset = TransformDataset(dataset, transform=transform_train)
    else:
        transform_test = transforms.Compose([
            RandomSample(6, 5),
            BatchLoad(),
            BatchApply(transforms.Resize(256)),
            BatchToTensor(),
        ])
        dataset = SRADDataset(root)
        dataset = TransformDataset(dataset, transform=transform_test)

    return dataset


def process_inputs(inputs, model, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []
    for _ in range(n_epochs):
        for x, y in inputs:
            x = x.to(device)
            x = model(x)
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            outputs.append((x, y))
            print(len(outputs))
    return outputs


def prepare_data(inputs, model, root, n_epochs=3):
    import numpy as np

    with torch.no_grad():
        d = process_inputs(inputs, model, n_epochs)

    x = [d for d, _ in d]
    y = [d.reshape(-1, 1) for _, d in d]
    np.save(os.path.join(root, "x.npy"), np.vstack(x))
    np.save(os.path.join(root, "y.npy"), np.vstack(y).reshape(-1))


class NumpyDataset(data.Dataset):

    def __init__(self, root):
        import numpy as np

        self.x = np.load(os.path.join(root, "x.npy"))
        self.y = np.load(os.path.join(root, "y.npy"))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return self.x.shape[0]


class ItemClass:

    def __init__(self, name, items):
        self.name = name
        self.items = items
        self.subclasses = []

    def add_sub_class(self, sub_class):
        self.subclasses.append(sub_class)

    def flatten(self, level=0, is_grouped=False):
        return ItemClass.__flatten__(self, level, "", is_grouped=is_grouped)

    @staticmethod
    def __flatten__(root_class, level, prefix, is_grouped):
        # result can be a parameter to reduce array creation
        result = []
        if level == 0:
            # prefix = "{}/{}".format(prefix, root_class.name)
            prefix = root_class.name
        if level <= 0:
            if is_grouped:
                result.append((root_class.items, prefix))
            else:
                for item in root_class.items:
                    result.append((item, prefix))
        for sub_class in root_class.subclasses:
            result += ItemClass.__flatten__(sub_class, level - 1, prefix, is_grouped)
        return result

    @staticmethod
    def item_class(root, extensions):
        from torchvision.datasets.folder import has_file_allowed_extension

        item_class = ItemClass(os.path.basename(root), [])
        for target in sorted(os.listdir(root)):
            d = os.path.join(root, target)
            if os.path.isdir(d):
                item_class.add_sub_class(ItemClass.item_class(d, extensions))
            else:
                if has_file_allowed_extension(d, extensions):
                    item_class.items.append(d)
        return item_class


class MovingMNIST(data.Dataset):
    def __init__(self, filename):
        import numpy as np

        self.mnist = np.load(filename).astype(np.float32)

    def __getitem__(self, index):
        return self.mnist[:, index:index + 1] / 255

    def __len__(self):
        return self.mnist.shape[1]


class SRADDataset(data.Dataset):

    def __init__(self, root):
        self.samples = SRADDataset.make_samples(root, 1)

    def get_id(self, index):
        seq, seq_id = self.samples[index]
        return seq_id

    def __getitem__(self, index):
        return self.samples[index][0]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def make_samples(root, level):
        from torchvision.datasets.folder import IMG_EXTENSIONS

        item_class = ItemClass.item_class(root, IMG_EXTENSIONS)
        return item_class.flatten(level=level, is_grouped=True)


class ImageDir(data.Dataset):

    def __init__(self, root, loader=None, level=0):
        from torchvision.datasets.folder import IMG_EXTENSIONS
        from torchvision.datasets.folder import default_loader

        if loader:
            self.loader = loader
        else:
            self.loader = default_loader

        self.samples = ImageDir.make_samples(root, IMG_EXTENSIONS, level=level)

    def __getitem__(self, index):
        path, y = self.samples[index]
        x = self.loader(path)
        return x, os.path.basename(path)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def make_samples(dir, extensions, level):
        item_class = ItemClass.item_class(dir, extensions)
        return item_class.flatten(level=level)


class TransformDataset(data.Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.transform:
            return self.transform(item)
        else:
            return item

    def __len__(self):
        return len(self.dataset)
