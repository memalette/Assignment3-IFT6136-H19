# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:24:34 2019

@author: Remi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import numpy as np

class VAE_q2(nn.Module):
    def __init__(self):
        super(VAE_q2, self).__init__()
        
        # Non-linearity
        self.elu          = nn.ELU()
        
        # Encoder
        self.en_conv1     = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.en_avgpool1  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.en_conv2     = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.en_avgpool2  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.en_conv3     = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5))
        self.en_linear1   = nn.Linear(in_features=256, out_features=100)
        self.en_linear2   = nn.Linear(in_features=256, out_features=100)

            
        # Decoder
        self.de_linear1 = nn.Linear(100,256)
        self.de_conv1   = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5),padding=4)
        self.de_conv2   = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3),padding=2)
        self.de_conv3   = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3),padding=2)
        self.de_conv4   = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3),padding=2)
        self.sig        = nn.Sigmoid()
    
    def UNFlatten(self,x):
        return x.reshape(x.shape[0],1,28,28)    

    def Flatten(self,x):
        return x.reshape(x.shape[0],784)
    
    def encoder(self, x): 
        x = self.UNFlatten(x)
        
        h = self.elu(self.en_conv1(x))
        h = self.en_avgpool1(h)
        h = self.elu(self.en_conv2(h))
        h = self.en_avgpool2(h)
        h = torch.squeeze(self.elu(self.en_conv3(h))) 
        
        return self.en_linear1(h), self.en_linear2(h)
    
    def decoder(self,z):
        
        z = self.elu(self.de_linear1(z))
        z = z.view(-1, 256, 1, 1)
        z = self.elu(self.de_conv1(z))
        z = nn.functional.interpolate(z,scale_factor=2,mode='bilinear')
        z = self.elu(self.de_conv2(z))
        z = nn.functional.interpolate(z,scale_factor=2,mode='bilinear')
        z = self.elu(self.de_conv3(z))
        z = self.de_conv4(z)
        return self.Flatten(self.sig(z))
    
    def reparametrization_trick(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = Variable(torch.randn(*mu.size()).cuda())
        
        z = mu + std * esp
        return z
        
    def forward(self, x):
        mus, log_vars = self.encoder(x)
        recon_x=self.decoder(self.reparametrization_trick(mus, log_vars))
        

        return recon_x, mus, log_vars

        
    def loss(self, x, recon_x, mus, log_vars):
        
        recon_log = F.binary_cross_entropy(recon_x, x, reduction='sum')        
        D_KL      = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())
        
        return (recon_log + D_KL)/x.shape[0], recon_log, D_KL
    
    

# In[]
class VAE_q3(nn.Module):
    def __init__(self):
        super(VAE_q3, self).__init__()
        
        #Non-linearity
        self.elu          = nn.ELU()
        # Encoder
        
        self.en_conv1     = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.en_avgpool1  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.en_conv2     = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.en_avgpool2  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.en_conv3     = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(6, 6))
        self.en_linear1_1   = nn.Linear(in_features=256, out_features=100)
        self.en_linear2_1   = nn.Linear(in_features=256, out_features=100)
        
        
        # Decoder
        self.de_linear1 = nn.Linear(100,256)
        self.de_conv1   = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=6,padding=0)
        self.up1        = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_conv2   = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3),padding=0)
        self.up2        = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_conv3   = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3),padding=0)
        self.de_conv4   = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3,padding=0)
        self.tanh       = nn.Tanh()
        
        # Standard deviation
        self.Splus      = nn.Softplus() 
        
        # loss
        self.MSE = nn.MSELoss(reduction='sum')
        
    def encoder(self, x): 
        
        h = self.elu(self.en_conv1(x))
        h = self.en_avgpool1(h)
        h = self.elu(self.en_conv2(h))
        h = self.en_avgpool2(h)
        h = torch.squeeze(self.elu(self.en_conv3(h))) 
        stds=self.Splus(self.en_linear2_1(h))+1e-5
        return self.en_linear1_1(h), stds
    
    def decoder(self,z):
        
        z = self.elu(self.de_linear1(z))
        z = z.view(-1, 256, 1, 1)
        
        z = self.elu(self.de_conv1(z))
        
        z = self.up1(z)
        z = self.elu(self.de_conv2(z))
        z = self.up2(z)
        z = self.elu(self.de_conv3(z))
        
        z = self.de_conv4(z)
        
        return self.tanh(z)
    
    def reparametrization_trick(self, mu, stds):
        # return torch.normal(mu, std)
        esp = Variable(torch.randn(*mu.size()).cuda())
        
        z = mu + stds * esp
        return z
        
    def forward(self, x):
        mus, stds = self.encoder(x)
        recon_x=self.decoder(self.reparametrization_trick(mus, stds))
        
        return recon_x, mus, stds
    
    def loss(self, x, recon_x, mus, stds):
        
        recon_loss = torch.sum((recon_x- x).pow(2))
        
        D_KL      = (-0.5 * torch.sum(1 + 2*stds.log() - mus.pow(2) - stds.pow(2)))
        
        return (recon_loss + D_KL)/x.shape[0],recon_loss,D_KL        