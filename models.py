import torch
import torch.nn.functional as F


class augmentor(torch.nn.Module):
    
    def __init__(self,
                gpu=True):
        
        super().__init__()
        
        self._set_device(gpu=gpu)
        
        self.c1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c2 = torch.nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c3 = torch.nn.Conv2d(in_channels=9, out_channels=27, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c4 = torch.nn.Conv2d(in_channels=27, out_channels=81, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c5 = torch.nn.Conv2d(in_channels=81, out_channels=81, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)
        
        self.cT5 = torch.nn.ConvTranspose2d(in_channels=81, out_channels=81, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.cT4 = torch.nn.ConvTranspose2d(in_channels=81, out_channels=27, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.cT3 = torch.nn.ConvTranspose2d(in_channels=27, out_channels=9, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.cT2 = torch.nn.ConvTranspose2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.cT1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)
        
        
        
        
    def _set_device(self, gpu=True):
        self.device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'; print(f'device: {self.device}')
    
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x0 = self.c4(x)
        x = self.c5(x0)
        
        x = self.cT5(x) + x0
        x = self.cT4(x)
        x = self.cT3(x)
        x = self.cT2(x)
        x = self.cT1(x)
        
        return x

class encoder(torch.nn.Module):
    
    def __init__(self,
                gpu=True):
        
        super().__init__()
        
        self._set_device(gpu=gpu)
        
        self.c1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c2 = torch.nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c3 = torch.nn.Conv2d(in_channels=9, out_channels=27, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c4 = torch.nn.Conv2d(in_channels=27, out_channels=81, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)        
        self.c5 = torch.nn.Conv2d(in_channels=81, out_channels=81, kernel_size=3, stride=1, padding=0, device=self.device, dtype=None)
        
        self.fc1 = torch.nn.Linear(26244, 50)
        
        
        
        
    def _set_device(self, gpu=True):
        self.device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'; print(f'device: {self.device}')
    
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = torch.nn.Flatten()(x)
        x = self.fc1(x)
        
        return x
        
        
        