import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import os
import torchvision
# --------
# Datasets
# --------


class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.,
        order = None,
        initial_class_num = 0
    ):  
        self.dataset_name = dataset_name
        datasets = _get_datasets(dataset_name)
        self._setup_data(
            datasets,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            generate_order=order,
            initial_class_num=initial_class_num
        )
        self.train_transforms = datasets[0].train_transforms  # FIXME handle multiple datasets
        self.test_transforms = datasets[0].test_transforms

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        
    @property
    def n_tasks(self):
        return len(self.increments)

    def new_task(self, memory=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        x_train, y_train = self._select(
            self.data_train, self.targets_train, low_range=min_class, high_range=max_class
        )
        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)
        index_train = np.zeros((len(x_train)), dtype=np.int)
        index_val = np.zeros((len(x_val)), dtype=np.int)
        index_test = np.zeros((len(x_test)), dtype=np.int)
        
        if memory is not None:
            data_memory, targets_memory = memory
            print("Set memory of size: {}.".format(data_memory.shape[0]))
            x_train = np.concatenate((x_train, data_memory))
            y_train = np.concatenate((y_train, targets_memory))
            index_train = np.concatenate((index_train, np.ones((len(data_memory)), dtype=np.int))) #0:new 1:old
        
        train_loader = self._get_loader(x_train, y_train,index_train, mode="train")
        val_loader = self._get_loader(x_val, y_val,index_val, mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test,index_test, mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def get_custom_loader(self, class_indexes, mode="test", data_source="train"):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)
        index = np.ones((len(data)), dtype=np.int)
        return data, self._get_loader(data, targets,index, shuffle=False, mode=mode)

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
    
    def _get_loader(self, x, y,index, shuffle=True, mode="train"):
        if mode == "train":
            trsf = transforms.Compose(self.train_transforms)
        elif mode == "test":
            trsf = transforms.Compose(self.test_transforms)
        elif mode == "flip":
            trsf = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=1.), *self.test_transforms]
            )
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))
        
        return DataLoader(
            DummyDataset(x, y,index, trsf, self.dataset_name),
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._workers
        )

    def _setup_data(self, datasets, random_order=False, seed=1, increment=10, validation_split=0., generate_order=None, initial_class_num=0):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            dataset.class_order = dataset._define_class_order(generate_order)
            x_train, y_train, x_val, y_val, x_test, y_test = dataset.data(validation_split)

            order = [i for i in range(len(np.unique(y_train)))]
            if random_order:
                random.seed(seed)  # Ensure that following order is determined by seed:
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order

            self.class_order.append(order)

            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:
                self.increments.append(len(order))
            else:
                self.increments = [initial_class_num]
                self.increments = self.increments + [increment for _ in range((len(order)-initial_class_num) // increment )]
            print(self.increments)
            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)


    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    
class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y,index, trsf, dataset_name):
        self.x, self.y, self.index = x, y,index
        self.trsf = trsf
        self.dataset_name = dataset_name

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y,index = self.x[idx], self.y[idx], self.index[idx]  
        if self.dataset_name == "imagenet":
            x = Image.open(x).convert('RGB')
        else:
            x = Image.fromarray(x)
        x = self.trsf(x)
        return x, y, index


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet":
        return iImageNet
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def _split_per_class(x, y, validation_split=0.):
        """Splits train data for a subset of validation data.

        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
        
        return x_val, y_val, x_train, y_train
    
class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = [transforms.ToTensor()]
    class_order = None
    
    def _define_class_order(order):
        order_0 = [53, 37, 65, 51, 4, 20, 38, 9, 10, 81, 44, 36, 84, 50, 96, 90, 66, 16, 80, 33, 24, 52, 91, 99, 64, 5, 58, 76, 39, 79, 23, 94, 30, 73, 25, 47, 31, 45, 19, 87, 42, 68, 95, 21, 7, 67, 46, 82, 11, 6, 41, 86, 88, 70, 18, 78, 71, 59, 43, 61, 22, 14, 35, 93, 56, 28, 98, 54, 27, 89, 1, 69, 74, 2, 85, 40, 13, 75, 29, 34, 92, 0,  77, 55, 49, 3, 62, 12, 26, 48, 83, 60, 57, 63, 15, 32, 8, 97, 72, 17]
        order_1 = [12, 27, 0, 55, 60, 16, 5, 90, 4, 52, 2, 44, 84, 49, 73, 56, 1, 59, 58, 75, 70, 41, 51, 6, 69, 94, 45, 33, 22, 40, 71, 72, 67, 86, 48, 28, 38, 3, 23, 79, 92, 82, 8, 30, 13, 21, 11, 24, 14, 9, 96, 15, 98, 91, 97, 53, 68, 47, 63, 85, 36, 31, 76, 77, 83, 78, 34, 88, 61, 64, 93, 43, 37, 18, 54, 87, 46, 74, 50, 65, 32, 66, 19, 29, 95, 57, 99, 62, 26, 89, 80, 25, 81, 10, 20, 39, 7, 42, 17, 35]   
        order_2 = [40, 79, 39, 74, 4, 95, 20, 28, 68, 22, 6, 61, 25, 17, 52, 44, 47, 1, 59, 31, 60, 58, 94, 56, 42, 36, 80, 92, 11, 69, 0, 62,14, 19, 37, 7, 81, 26, 5, 75, 3, 48, 24, 54, 88, 66, 53, 64, 65, 12, 57, 87, 91, 30, 13, 10, 45, 89, 82, 49, 99, 18, 96, 27,23, 35, 9, 76, 77, 29, 55, 90, 84, 46, 83, 43, 97, 72, 67, 78, 63, 70, 15, 73, 8, 51, 33, 86, 85, 21, 41, 38, 32, 2, 93, 71, 16, 50, 34, 98]
        #easy order
        order_3 = [4 ,31 ,55 ,72 ,95 ,1 ,33 ,67 ,73 ,91 ,54 ,62 ,70 ,82 ,92 ,9 ,10 ,16 ,29 ,61 ,0 ,51 ,53 ,57 ,83 ,22 ,25 ,40 ,86 ,87 ,5 ,20 ,26 ,84 ,94 ,6 ,7 ,14 ,18 ,24 ,3 ,42 ,43 ,88 ,97 ,12 ,17 ,38 ,68 ,76 ,23 ,34 ,49 ,60 ,71 ,15 ,19 ,21 ,32 ,39 ,35 ,63 ,64 ,66 ,75 ,27 ,45 ,77 ,79 ,99 ,2 ,11 ,36 ,46 ,98 ,28 ,30 ,44 ,78 ,93 ,37 ,50 ,65 ,74 ,80 ,47 ,52 ,56 ,59 ,96 ,8 ,13 ,48 ,58 ,90 ,41 ,69 ,81 ,85 ,89]
        #hard_order
        order_4 = [4, 54, 0, 5, 3, 23, 35, 2, 37, 8, 31, 62, 51, 20, 42, 34, 63, 11, 50, 13, 55, 70, 53, 26, 43, 49, 64, 36, 65, 48, 72, 82, 57, 84, 88, 60, 66, 46, 74, 58, 95, 92, 83, 94, 97, 71, 75, 98, 80, 90, 1, 9, 22, 6, 12, 15, 27, 28, 47, 41, 33, 10, 25, 7, 17, 19, 45, 30, 52, 69, 67, 16, 40, 14, 38, 21, 77, 44, 56, 81, 73, 29, 86, 18, 68, 32, 79, 78, 59, 85, 91, 61, 87, 24, 76, 39, 99, 93, 96, 89]
        if order == 0 :
            class_order = order_0
        elif order == 1:
            class_order = order_1
        elif order == 2:
            class_order = order_2
        elif order == 3:
            class_order = order_3
        elif order == 4:
            class_order = order_4
        elif order == 5:
            class_order = order_5
        return class_order    
    
class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(iCIFAR10):
    #base_dataset = datasets.cifar.CIFAR100
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]
    
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    def data(validation_split=0):
        train_dataset = datasets.cifar.CIFAR100("data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("data", train=False, download=True)
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_val, y_val, x_train, y_train = _split_per_class(
            x_train, y_train, validation_split
        )
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
            
        return x_train, y_train, x_val, y_val, x_test, y_test
    
class iImageNet(DataHandler):    
    #base_dataset = datasets.ImageNet
    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    
    def data(validation_split=0.):
        traindir = os.path.join("../data/seed_1993_subset_100_imagenet/data", 'train')
        valdir = os.path.join("../data/seed_1993_subset_100_imagenet/data", 'val')

        trainset = datasets.ImageFolder(traindir)
        evalset =  datasets.ImageFolder(valdir)
        
        x_train, y_train = split_images_labels(trainset.imgs)
        x_test, y_test = split_images_labels(evalset.imgs)
        
        
        x_train, y_train = x_train, np.array(y_train)
        x_val, y_val, x_train, y_train = _split_per_class(
                x_train, y_train, validation_split
            )
        x_test, y_test = x_test, np.array(y_test)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    

class iMNIST(DataHandler):
    base_dataset = datasets.MNIST
    train_transforms = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()]
    test_transforms = [transforms.ToTensor()]

    

class iPermutedMNIST(iMNIST):

    def _preprocess_initial_data(self, data):
        b, w, h, c = data.shape
        data = data.reshape(b, -1, c)

        permutation = np.random.permutation(w * h)

        data = data[:, permutation, :]

        return data.reshape(b, w, h, c)


# --------------
# Data utilities
# --------------
