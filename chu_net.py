import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from base_net import Base_Net

class CHU_Net(Base_Net):
    '''
    ##########################################################################################
    ### Learning method devised in "Unsupervised Learning by Competing Hidden Units" (CHU) ###
    ##########################################################################################
        - ref: https://www.pnas.org/content/pnas/116/16/7723.full.pdf
    
    TODO:
        - model tops
        - loss options
            - structure as optim and optim_dict
        - support for multiple hidden layers
        
    Goals (not implemented): 
        - modular support for different neural coding and local learning schemes
    
    IDEA BOX:
        - learned hebbian/anti-hebbian strengths
        - decaying/increasing anti-hebbian strength
        - learned convolutional kernels
    '''
    
    def __init__(self, gpu=True):
        super().__init__(gpu=gpu)
        
    def load_data(self,
                  data,
                  synapse_shape=None,
                  n_classes=None,
                  lr_s=1e-1,
                  lr_t=8e-2,
                  lr_decay_s=0.95,
                  lr_decay_t=0.97,
                  momentum=0.5,
                  conv=None,
                  reset=True,
                  verbose=False,
                 ):
        '''
        Uses parent class to load data and inits weights. 
        
        - Args
            data (data.py/Datagen OR numpy array):
                data.py/Datagen or np arr of square images
            lr_s/lr_t/lr_decay_s/lr_decay_t (float):
                Learning rates and decay rates for model synpases and tops (output layers) respectively
        '''
        super().load_data(data,
                          reset=reset,
                          verbose=verbose)
        
        if reset:
            self.synapse_shape = synapse_shape
            self.n_hidden = len(self.synapse_shape)
            self.lr_s = lr_s
            self.lr_t = lr_t
            self.lr_decay_s = lr_decay_s
            self.lr_decay_t = lr_decay_t
            self.conv = conv.to(self.device) if conv else None

            if n_classes:
                self.fc = torch.nn.Linear(self.synapse_shape[-1], n_classes).to(self.device)
                self.opt = optim.Adam(self.parameters(),)
#                                       lr=lr_t)
#                 self.opt = optim.SGD(self.parameters(),
#                                       lr=lr_t,
#                                       momentum=momentum)
                
            if self.conv:
                self.conv_dims = self._get_conv_dims()
                synapse_length = np.prod(self.conv_dims)
                
                if len(self.train_shape) == 2:
                    c = self.train_shape[0]
                    s = int(self.train_shape[1] ** 0.5)
                    self.train_data = torch.reshape(self.train_data, (c, 1, s, s))
            else:
                synapse_length = np.prod(self.train_shape[1:])
            if synapse_shape:
                self.synapses = [torch.Tensor(n, synapse_length).normal_(0, 1).to(self.device) for n in self.synapse_shape]
    
    def _get_conv_dims(self):
        '''
        Returns (tuple): 
            (n_channels, side_length^2)
        '''
        # side length
        l = np.prod(self.train_shape[1:]) ** 0.5
        # padding 
        p = self.conv.padding[0]
        # kernel size
        k = self.conv.kernel_size[0]
        # stride
        s = self.conv.stride[0]
        return (self.conv.out_channels, int(np.floor((l + 2*p - (k-1) - 1)/s + 1) ** 2))
    
    def grid(self, data, nx, ny, title=None):
        if self.conv:
            w = int(self.conv_dims[1] ** 0.5)
            h = int(self.conv_dims[0] * w)
        else:
            w = h = int(np.prod(self.train_shape[1:]) ** 0.5)
            
        super().grid(data, nx, ny, h=h, w=w, title=title)
    
    def train_synapses(self,
                       epochs,
                       bs=None,
                       prec=1e-30,
                       delta=0.4,
                       p=2.0,
                       k=2,
                       update_freq=None,
                      ):
        '''
        - Args
            lr (float):
                Learning rate
            bs (int):
                Batch size
            prec (float):
                Normalization term used to avoid exploding updates
            delta (float):
                Strength of anti-Hebbian learning. Very sensitive
            p (int):
                The Lebesgue p-norm of the weights
            k (tuple(int,int)):
                Number of synapses to experience hebbian, anti-hebbian learning
        '''
        
        history = {'times': []}
        
        if self.train_form == 'gen':
            bs = self.train_gen.bs
        
        for epoch in range(self.epoch, self.epoch+epochs):
            t0 = time.time()
            if update_freq and not epoch % update_freq:
                print(f'Epoch #{epoch}')
                
            self._shuffle()
            lr = self.lr_s * (self.lr_decay_s ** self.epoch) if self.lr_decay_s else self.lr_s
#             lr = self.lr_s * (1 - (self.epoch-epoch)/(epochs))
            
            for batch, _ in self._get_batch(bs=bs):

                if self.conv:       
                    with torch.no_grad():
                        batch = self.conv(batch)
                    
                if len(batch.size()) > 2:
                    batch = batch.flatten(start_dim=1)
                
                for i in range(self.n_hidden+1):
                
                    # Dot the synapses with the batch inputs for the currents (Ii, Fig. 2)
                    sig = self.synapses[i].sign().to(self.device)
                    currents = (sig * self.synapses[i].abs().to(self.device) ** (p-1)).matmul(batch.t())

                    # Score the currents to find activation function g(i), Eq. 10
                    _, indices = currents.topk(k[0]+k[1], dim=0)
                    best_ind, best_k_inds = indices[:k[0]], indices[k[0]:k[0]+k[1]]

                    g = torch.zeros(self.n_synapses, bs).to(self.device)
                    g[best_ind, torch.arange(bs).to(self.device)] = 1.0
                    g[best_k_inds, torch.arange(bs).to(self.device)] = -delta

                    # Use the sums of the scores to weight the synapse adjustments
                    synapse_scores = (g * currents).sum(dim=1)
                    ds = torch.matmul(g, batch) - synapse_scores.unsqueeze(1) * self.synapses[i]

                    update = lr * ds / max(ds.abs().max(), prec)
                    self.synapses[i] += update
                    
                    if i <= self.n_hidden: batch = self.synapses[i] * batch.t()

            self.epoch += 1
            history['times'].append(time.time()-t0)
        
        return history    
    
    def train_top(self,
                  epochs,
                  bs,
                  val=True,
                  train_conv=False,
                  update_freq=50):
        
        if self.conv and not train_conv:
            for p in [*self.parameters()][:2]:
                p.requires_grad = False
        
        self.train()
        
        n_train = self.train_shape[0]
        history = {'train_loss': np.zeros(epochs)}
        if val:
            history['val_loss'] = np.zeros(epochs)
            history['val_acc'] = np.zeros(epochs)
        
        for epoch in range(1, epochs+1):
            self.opt.defaults['lr'] = self.lr_t * (self.lr_decay_t ** (epoch-1)) \
                                        if self.lr_decay_t else self.lr_t
            
            for i, (batch, labels) in enumerate(self._get_batch(bs=bs), 1):
                self.opt.zero_grad()
                output = self.forward(batch)
                loss = F.nll_loss(output, labels)
                loss.backward()
                self.opt.step()
                history['train_loss'][epoch-1] += loss.item()
            
            history['train_loss'][epoch-1] /= n_train/bs
            train_loss = history['train_loss'][epoch-1]    
            
            if val:
                val_loss, val_acc = self.test(bs=bs)
                history['val_loss'][epoch-1], history['val_acc'][epoch-1] = val_loss, val_acc
                
            if not epoch % update_freq:
                print(f'EPOCH #{epoch}\tloss: {train_loss:.5f}\t' + \
                      f'val_loss: {val_loss:.5f}\tval_acc: {val_acc:.5f}' if val else None)
        
        return history
                
    def test(self,
             bs,             
             ):
        '''
        TODO
        '''
        self.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for i, (batch, labels) in enumerate(self._get_batch(bs=bs, train=False)):
                output = self.forward(batch)
                test_loss += F.nll_loss(output, labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()
        
        n_test = self.test_shape[0]
        
        return bs*test_loss/n_test, 100.*correct.item()/n_test
    
    def forward(self,
                x):      
        if self.conv:
            x = self.conv(x).flatten(start_dim=1)
        for s in self.synapses:
            x = x.matmul(s.t())
        x = self.fc(x)
        return F.log_softmax(x)
    
    def _get_batch(self,
                   bs=None,
                   train=True):
        yield from ((i, labels) if self.conv else (i.flatten(start_dim=1), labels) for (i, labels) in super()._get_batch(bs=bs, train=train))
        

class CHU_Ctrl(CHU_Net):
    '''
    Dummy NN to benchmark against CHU_Net
    '''
    def __init__(self,
                 gpu=True,
                 hidden=True,
                 conv=False):
        
        super().__init__(gpu=gpu)
        
        self.hidden = hidden
        self.conv = conv
        
        if self.conv:
            self.c1 = self.conv.to(self.device)
            if self.hidden:
                self.fc1 = torch.nn.Linear(10140, 2000).to(self.device)
                self.fc2 = torch.nn.Linear(2000, 10).to(self.device)
            else:
                self.fc1 = torch.nn.Linear(784, 10).to(self.device)
        else:
            if self.hidden:
                self.fc1 = torch.nn.Linear(784, 2000).to(self.device)
                self.fc2 = torch.nn.Linear(2000, 10).to(self.device)
            else:
                self.fc1 = torch.nn.Linear(784, 10).to(self.device)
    
    def load_data(self,
                  data,
                  lr=8e-2,
                  lr_decay=0.97,
                  momentum=0.5,
                  reset=True,
                  verbose=False,
                 ):
        
        super().load_data(data,
                          lr_t=lr,
                          lr_decay_t=lr_decay,
                          momentum=momentum,
                          reset=reset,
                          verbose=verbose)
        
        if reset:
            self.lr = lr
            self.lr_decay = lr_decay
            self.opt = optim.Adam(self.parameters(),
                                  lr)
    
    def forward(self, x):
        if self.conv:
            x = self.c1(x)
        x = F.relu(self.fc1(x))
        
        if self.hidden:
            x = self.fc2(x)
            
        return F.log_softmax(x)