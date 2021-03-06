import torch
import torch.nn.functional as F
import torchvision
import os


class base(torch.nn.Module):
    
    def __init__(self,
                 shape=None,
                 gpu=True):
        
        super().__init__()
        
        self.shape = shape
        self.n_channels = self.shape[0]
        
        self._set_device(gpu=gpu)
    

    def _set_device(self, gpu=True):
        self.device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'; print(f'device: {self.device}')

           
           
class joint(base):
    
    def __init__(self,
                 augmentation_comparator='pool',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.e = encoder(**kwargs)
        
        self.a = augmentor(comparator=augmentation_comparator,
                           **kwargs)
        
        
    def forward(self,
               x):
           
        xe = self.e(x)
        xa = self.a(x)
        
        return (self.e(x), self.a(x))
    
    
    def evaluate(self,
                 x):
        
        # returns
        # ( Xe, self.augmentor_loss(xe1, Xe1) ),
        # ( xa1, Xa1, self.encoder_loss(xa1, Xa1) )
        
        return self.e.evaluate_encoder(x, self.a(x)), self.a.evaluate_augmentor(x)
    
    
           
class encoder(base):
    
    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.encoder_loss = torch.nn.MSELoss()
        
        self.c1 = torch.nn.Conv2d(in_channels=1*self.n_channels, out_channels=3*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c2 = torch.nn.Conv2d(in_channels=3*self.n_channels, out_channels=9*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c3 = torch.nn.Conv2d(in_channels=9*self.n_channels, out_channels=27*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c4 = torch.nn.Conv2d(in_channels=27*self.n_channels, out_channels=27*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)
        
        self.fc1 = torch.nn.Linear((self.shape[-1] - 8)**2 \
                                       * 27
                                       * self.n_channels,
                                   1000)
        self.fc2 = torch.nn.Linear(1000, 50)
        
        self.act1 = torch.nn.Sigmoid()
        self.act2 = torch.nn.ReLU()
    
    
    def forward(self,
                x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = torch.nn.Flatten()(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        
        return x
        
    
    def evaluate_encoder(self,
                         x,
                         X):
        
        x1 = self.forward(x)
        X1 = self.forward(X)
        
        return x1, X1, self.encoder_loss(x1, X1) 

        
        
class augmentor(base):
    
    def __init__(self,
                 augmentation_comparator='pool',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.augmentor_loss = torch.nn.MSELoss()
        
        if augmentation_comparator == 'cnn':
            self.comparator_model = torchvision.models.inception_v3()

            if not os.path.exists('inception_v3_google-0cc3c7bd.pth'):
                print('DOWNLOAD MODEL PARAMETERS AT:\nhttps://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth')

            self.comparator_model.load_state_dict(torch.load('inception_v3_google-0cc3c7bd.pth'))
            self.comparator_model.eval()
        
        
        self.c1 = torch.nn.Conv2d(in_channels=1*self.n_channels, out_channels=3*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c2 = torch.nn.Conv2d(in_channels=3*self.n_channels, out_channels=9*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c3 = torch.nn.Conv2d(in_channels=9*self.n_channels, out_channels=27*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c4 = torch.nn.Conv2d(in_channels=27*self.n_channels, out_channels=27*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)
        
        fc_dim = (self.shape[-1]-12) \
                        ** 2 \
                        * self.n_channels \
                        * 27
        self.fc1 = torch.nn.Linear(fc_dim, fc_dim)
        
        self.cT4 = torch.nn.ConvTranspose2d(in_channels=27*self.n_channels, out_channels=27*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None) 
        self.cT3 = torch.nn.ConvTranspose2d(in_channels=27*self.n_channels, out_channels=9*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None) 
        self.cT2 = torch.nn.ConvTranspose2d(in_channels=9*self.n_channels, out_channels=3*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)   
        self.cT1 = torch.nn.ConvTranspose2d(in_channels=3*self.n_channels, out_channels=1*self.n_channels, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)
        
        self.act1 = torch.nn.Sigmoid()
    
    
    def forward(self,
                x):
        x = self.c1(x)
        x = self.c2(x)
        x0 = self.c3(x)
        x = self.c4(x0)
        x1 = self.c4(x)
        x = self.c4(x1)
        
        in_shape = x.shape
        
        x = torch.nn.Flatten()(x)
        x = self.act1(self.fc1(x))
        x = x.reshape(in_shape)
        
        x = self.cT4(x) + x1
        x = self.cT4(x)
        x = self.cT4(x) + x0
        x = self.cT3(x)
        x = self.cT2(x)
        x = self.cT1(x)
        
        return x
    
    
    def evaluate_augmentor(self,
                           x):
        
        A = self.forward(x).detach()
        
        loss = self.comparator(x, A)

        return A, loss
    
    
    def cnn_comparator(self,
                       x,
                       A):
        
        x = x.expand(-1, 3, -1, -1).resize_((x.shape[0],3,299,299))
        A = XA.expand(-1, 3, -1, -1).resize_((x.shape[0],3,299,299))
            
        x_out = self.comparator_model(x)
        A_out = self.comparator_model(X)
        
        return self.augmentor_loss(x_out, A_out)
    
    
    def pool_comparator(self,
                        x,
                        A):
        '''
        x_out = pool(x)
        A_out = pool(A)
        
        return self.augmentor_loss(x_out, A_out)
        
        '''
        
        pass
        
        

        
class baseline(torch.nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x