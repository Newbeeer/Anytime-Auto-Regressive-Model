import csv
import os

import torch.utils.data as data
from PIL import Image
from torchvision import transforms, datasets


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Borrowed from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MiniImagenet(data.Dataset):
    base_folder = '/data/lisa/data/miniimagenet'
    filename = 'miniimagenet.zip'
    splits = {
        'train': 'train.csv',
        'valid': 'val.csv',
        'test': 'test.csv'
    }

    def __init__(self, root, train=False, valid=False, test=False,
                 transform=None, target_transform=None, download=False):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.train = train
        self.valid = valid
        self.test = test
        self.transform = transform
        self.target_transform = target_transform

        if not (((train ^ valid ^ test) ^ (train & valid & test))):
            raise ValueError('One and only one of `train`, `valid` or `test` '
                             'must be True (train={0}, valid={1}, test={2}).'.format(train,
                                                                                     valid, test))

        self.image_folder = os.path.join(os.path.expanduser(root), 'images')
        if train:
            split = self.splits['train']
        elif valid:
            split = self.splits['valid']
        elif test:
            split = self.splits['test']
        else:
            raise ValueError('Unknown split.')
        self.split_filename = os.path.join(os.path.expanduser(root), split)
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use `download=True` '
                               'to download it')

        # Extract filenames and labels
        self._data = []
        with open(self.split_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for line in reader:
                self._data.append(tuple(line))
        self._fit_label_encoding()

    def __getitem__(self, index):
        filename, label = self._data[index]
        image = pil_loader(os.path.join(self.image_folder, filename))
        label = self._label_encoder[label]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def _fit_label_encoding(self):
        _, labels = zip(*self._data)
        unique_labels = set(labels)
        self._label_encoder = dict((label, idx)
                                   for (idx, label) in enumerate(unique_labels))

    def _check_exists(self):
        return (os.path.exists(self.image_folder)
                and os.path.exists(self.split_filename))

    def download(self):
        from shutil import copyfile
        from zipfile import ZipFile

        # If the image folder already exists, break
        if self._check_exists():
            return True

        # Create folder if it does not exist
        root = os.path.expanduser(self.root)
        if not os.path.exists(root):
            os.makedirs(root)

        # Copy the file to root
        path_source = os.path.join(self.base_folder, self.filename)
        path_dest = os.path.join(root, self.filename)
        print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
        copyfile(path_source, path_dest)

        # Extract the dataset
        print('Extract files from `{0}`...'.format(path_dest))
        with ZipFile(path_dest, 'r') as f:
            f.extractall(root)

        # Copy CSV files
        for split in self.splits:
            path_source = os.path.join(self.base_folder, self.splits[split])
            path_dest = os.path.join(root, self.splits[split])
            print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
            copyfile(path_source, path_dest)
        print('Done!')

    def __len__(self):
        return len(self._data)


def load_data(dataset_name, data_path):
    if dataset_name in ['mnist', 'fashion-mnist', 'cifar10']:
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_path, train=False, transform=transform)
            num_channels = 1
        elif dataset_name == 'fashion-mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(data_path, train=False, transform=transform)
            num_channels = 1
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_path, train=False, transform=transform)
            num_channels = 3
        else:
            raise ValueError(dataset_name)
        valid_dataset = test_dataset
    elif dataset_name == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = MiniImagenet(data_path, train=True, download=True, transform=transform)
        valid_dataset = MiniImagenet(data_path, valid=True, download=True, transform=transform)
        test_dataset = MiniImagenet(data_path, test=True, download=True, transform=transform)
        num_channels = 3
    elif dataset_name == 'celeba':
        image_size = 64
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.CenterCrop(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Define the train & test datasets
        train_dataset = datasets.CelebA(root=data_path,
                                        split='train', target_type="", download=True, transform=transform)
        test_dataset = datasets.CelebA(root=data_path,
                                       split='test', target_type="", download=True, transform=transform)
        num_channels = 3
        valid_dataset = test_dataset

    else:
        raise ValueError()

    return train_dataset, valid_dataset, test_dataset, num_channels
