from PIL import Image
import os
import os.path
import numpy as np
import pickle

import torch
import torchvision
from torchvision.datasets.vision import VisionDataset


# Default dir to house datasets
LOCAL_DATA_DIR = os.path.join(os.getcwd(), 'common_datasets')


class Datagen():
    '''
    Pytorch base class DataLoader wrapper for ease of use with common datasets.
        - Applies normalization transforms using global mean/stddev
        - Yields Torch tensors
    
    - Usage
        #################################
        from data import mnist_gen
        
        mgen = mnist_gen(bs=100)
        batch_idx, (batch, labels) = next(mnist_gen.generator)
        #################################
    
    TODO:
        - Helper functions to create Pytorch Dataset objects from np arrs
        - 
        
    '''
    def __init__(self,
                 bs,
                 base_dir=None):
        
        self.bs = bs
        self.base_dir = base_dir

    
    def _set_dataset(self,
                     dataset,
                     sample_shape = None,
                     transforms=None,):
        
        self.dataset = dataset
        
        self.transforms = transforms
        self.train_shape = tuple([(len(self.generator(train=True, download=True))) * self.bs]) + (sample_shape)
        self.test_shape = tuple([(len(self.generator(train=False))) * self.bs]) + (sample_shape)
        
    
    def generator(self,
                  train=True,
                  bs=None,
                  download=False,
                 ):
        
        loader = torch.utils.data.DataLoader(
                            self.dataset(LOCAL_DATA_DIR,
                                        train=train,
                                        download=download,
                                        transform=self.transforms),
                            batch_size=self.bs if bs is None else bs,
                            shuffle=False,
                            drop_last=True)
        return loader
        
    

class MNIST_gen(Datagen):
    def __init__(self,
                 bs):
        
        super(MNIST_gen, self).__init__(bs)
        self._set_dataset(dataset=torchvision.datasets.MNIST,
                          transforms=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                          ]),
                          sample_shape=(1,28,28))
        
        self.classes = ['zero', 'one', 'two', 'three', 'four', 
                        'five', 'six', 'seven', 'eight', 'nine']
        
        
        
class CIFAR_gen(Datagen):
    '''
    TODO:
        - Normalize w/global mean/stddev
    '''
    def __init__(self,
                 bs,):
        
        super(CIFAR_gen, self).__init__(bs)
        self._set_dataset(CIFAR10_custom,
                          transforms=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                          ]),
                          sample_shape=(3,32,32))
        
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        


class Custom_gen(Datagen):
    '''
    General use datagen for Pytorch Dataset object.
    '''
    def __init__(self,
                 bs,
                 dataset,
                 sample_shape,
                 transforms: list=None):
        
        super(Custom_gen, self).__init__(bs)
        self._set_dataset(dataset,
                          transforms=torchvision.transforms.Compose(transforms),
                          sample_shape=sample_shape)
        
        
        
        
class CIFAR10_custom(VisionDataset):
    
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10_custom, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")