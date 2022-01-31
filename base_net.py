import matplotlib.pyplot as plt
import numpy as np

import torch

class Base_Net(torch.nn.Module):
    '''
    ########################################################
    ### Pytorch base class for neuronal-learning testbed ###
    ########################################################
    
        - Modular prototyping of:
            - neuronal behaviors, 
            - connectivity schemes, 
            - network topologies, 
            - data representation
        - Currently supports:
            - learning method devised in "Unsupervised Learning by Competing Hidden Units" (CHU)
                - ref: https://www.pnas.org/content/pnas/116/16/7723.full.pdf
            - image classification with word2vec label representations
    
    TODO:
        - timing
        - logging
        - docstrings
        
    IDEA BOX:
    - Torch tensor -> np array helper function
    '''
    
    def __init__(self, gpu=True):
        super().__init__()
        
        self._set_device(gpu=gpu)
    
    def _set_device(self, gpu=True):
        self.device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'; print(f'device: {self.device}')
    
    def _shuffle(self):
        '''
        Shuffles in-memory data or resets datagen (assumes datagen shuffles on reset)
        '''
        if self.train_form == 'mem':
            perm = torch.randperm(self.train_data.size(0))
            self.train_data = self.train_data[perm]
            if self.train_labels:
                self.train_labels = self.train_labels[perm]
    
    def grid(self, data, nx, ny, h=28, w=28, title=None):
        '''
        Grid visualization for activations, etc.
        
        - Args
            data (numpy array):
                Accepts arrays of shape=(n_samples, dim_1, dim_2)
            nx (int):
                Number of samples in x direction
            ny (int):
                Number of samples in y direction    
            h, w (int):
                Sample height, width
            title (str):
                Plot title.
        '''
        plt.clf()
        fig = plt.figure()
        y_pos = 0
        height = h
        width = w
        canvas = np.zeros((height*ny, width*nx))
        
        for y in range(ny):
            for x in range(nx):
                canvas[y*height:(y+1)*height,x*width:(x+1)*width] = data[y_pos,:].reshape(height,width)
                y_pos += 1
        
        plt.clf()
        c_max = np.amax(np.absolute(canvas))
        if title: plt.title(title, fontsize=16)
        im = plt.imshow(canvas, cmap='bone', vmin=-c_max, vmax=c_max)
        fig.colorbar(im,ticks=[np.amin(canvas), 0, np.amax(canvas)])
        plt.yticks(range(0, ny*height, height), range(ny))
        plt.xticks(range(0, nx*width, width), range(nx))
        fig.canvas.draw()
    
    def load_data(self,
                  data,
                  labels=None,
                  test=False,
                  reset=True,
                  callbacks=None,
                  verbose=False):
        '''
        Supports 1D and 2D (grayscale/single channel) data types. Uses shallow copy.
        
        - Args
            data (array-like OR datagen from data.py:
                Assigns data form as in-memory (array-like) or as generator (datagen)
            labels (array-like):
                Datagens already include labels. Only needed for array-like data. 
            test (bool):
                Assigns data or datagen to train/test sets. Only needed for array-like data. 
            
        TODO:
            - Testing for labels
            - support for data->datagen conversion
            -
        '''
        
        self.callbacks = callbacks
        
        if reset:
            self.epoch = 0
        
        # array-like
        if hasattr(data, '__iter__'):
            data = torch.Tensor(data).to(self.device)
            _new_data = ('mem',
                        data,
                        tuple(data.size()),
                        torch.Tensor(labels).to(self.device) if labels else None)
            if not test:
                self.train_form, self.train_data, self.train_shape, self.train_labels = _new_data
            else:
                self.test_form, self.test_data, self.test_shape, self.test_labels = _new_data
            if verbose:
                dataset = self.train_data if not test else self.test_data
                print(f'Found {dataset.size()[0]} samples from {dataset.size()}\n')
        
        # generator
        elif hasattr(data, 'generator'):
            self.train_form, self.test_form, self.gen, self.train_shape, self.test_shape = ('gen',
                                                                                     'gen',
                                                                                     data,
                                                                                     data.train_shape,
                                                                                     data.test_shape,)
            if verbose:
                shape = self.train_shape if not test else self.test_shape
                print(f'Found {shape[0]} samples from {shape}')

        else:
            raise NotImplementedError('data can be array-like or Datagen from data.py')
    
    def _get_batch(self,
                    bs=None,
                    train=True):
        '''
        Generator for retrieving batches of data from data/datagen
        
        - Usage
            ########################################
            net = CHU_Net
            net.load_datagen(datagen) # or load_data
            for batch, labels in net._get_batch():
                do()...
            ########################################
        '''

        form = self.train_form if train else self.test_form

        if form == 'mem':
            assert type(bs) == int and bs>=1, 'Include batch size for in-memory data'

            data, labels, shape = (self.train_data, self.train_labels, self.train_shape) if train else (self.test_data, self.test_labels, self.test_shape)
            for i in range(shape[0]//bs):
                yield data[i*bs:(i+1)*bs], (labels[i*bs:(i+1)*bs] if labels else None)
        
        elif form == 'gen':
            yield from ((data.to(self.device), label.to(self.device)) for _,(data,label) in enumerate(self.gen.generator(train=train)))