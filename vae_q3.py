# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:45:39 2019

@author: Remi
"""

#!/usr/bin/env python
import torch

import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
import numpy as np
from VAEs import VAE_q3
import torch.optim as optim
from VAEs import get_data_loader
import torch.utils.data
import torchvision.datasets

# In[]
train, valid, test = get_data_loader("svhn", 64)

# In[]

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
    cuda_available=True
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
    of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")
    cuda_available=False
    

        
# In[Train the model]  
def train(epochs, l_r):    
    
    model = VAE_q3()
    
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda") 
        cuda_available=True
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
        of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")
        cuda_available=False    
        
    model.to(device)
    # We use ADAM with a learning rate of 3*10^-4
    optimizer = optim.Adam(model.parameters(), lr=l_r)        
    
    losses_train=[]
    losses_valid=[]
    loss_train = 0
    loss_valid = 0
    for epoch in range(epochs):
        prev_loss_train = loss_train
        prev_loss_valid = loss_valid
        loss_train = 0
        loss_valid = 0
        
        
        model.train()
        
        for batch_idx, (inputs, _ ) in enumerate(train):
            if cuda_available:
                inputs = inputs.cuda()
            
            optimizer.zero_grad()
            recon_x, mus, log_vars = model(inputs)
            
            loss, recon_loss, D_KL = model.loss(inputs, recon_x, mus, log_vars)
    
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
            #if batch_idx%50==0:
            print('Epoch : %d Train Loss: %d Recon Loss: %d DKL Loss: %d Prev Loss_train %d Prev Loss_valid %d Progress : %.3f ' % (epoch, loss.item(),recon_loss/inputs.shape[0], D_KL/inputs.shape[0], prev_loss_train,prev_loss_valid, batch_idx/len(train)))
        
        loss_train=loss_train/len(train)
        losses_train.append(loss_train)
        # Evaluate
        
        model.eval()
        
        best_val_so_far=1000
        for batch_idx, (inputs, _) in enumerate(valid):
            if cuda_available:
                inputs = inputs.cuda()
    
            recon_x, mus, log_vars = model(inputs)
            loss, recon_loss, D_KL = model.loss(inputs, recon_x, mus, log_vars)
            loss_valid += loss.item()
            
        loss_valid = loss_valid/len(valid)
        print('Epoch : %d Valid Loss : %.3f ' % (epoch, loss_valid))
        losses_valid.append(loss_valid)
        
        #Saving model if best
        if loss_valid < best_val_so_far:
            best_val_so_far = loss_valid
            print("Saving model parameters to best_params.pt")
            torch.save(model.state_dict(), os.path.join(os.getcwd(),'best_model_vae_q3', 'best_params_0418.pt'))
            
        model.train()

        return losses_train, losses_valid

# In[]

model=VAE_q3()
model.cuda()
path=os.path.join(os.getcwd(),'best_model_vae_q3', 'best_params_0418.pt')
model.load_state_dict(torch.load(path)) 


# In[Question 1]


norm_tensor_to_image = transforms.Compose([
    transforms.Normalize((-1., -1., -1.),
                         (2., 2., 2.)),
    transforms.ToPILImage()])

# Sample generation

w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    print(i)
    sample=model.decoder(torch.randn(100).cuda())  
    img = norm_tensor_to_image(sample[0, :, :, :].detach().cpu())
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

# In[Question 2]


torch.manual_seed(7778)
z = torch.randn(100).cuda()
epsilon = 2
vec_eps = np.zeros(shape=(1, 100, 1, 1))
x = model.decoder(z).detach().cpu()
x = norm_tensor_to_image(x[0, :, :, :])
plt.imshow(x)
plt.show()
    
fig = plt.figure(figsize=(20, 20))
    
for i in range(100):
    z_prime = z
    z_prime[i] = z[i] + torch.randn(1)*2
    x = model.decoder(z_prime)
    x = norm_tensor_to_image(x[0, :, :, :].detach().cpu())
    fig.add_subplot(10, 10, i+1)
    plt.imshow(x)
plt.show()



# In[Question 3]


torch.manual_seed(7778)
z1 = torch.randn(100).cuda()

torch.manual_seed(458)
z2 = torch.randn(100).cuda()

 
fig = plt.figure(figsize=(20, 20))
    
for i in range(10):
    z_prime = z1*(10-i)/10+z2*i/10
    x = model.decoder(z_prime)
    x = norm_tensor_to_image(x[0, :, :, :].detach().cpu())
    fig.add_subplot(1, 10, i+1)
    plt.imshow(x)
plt.show()


fig = plt.figure(figsize=(20, 20))
    
for i in range(10):
    x1 = model.decoder(z1)
    x2 = model.decoder(z2)
    x  = x1*(10-i)/10+x2*i/10 
    x = norm_tensor_to_image(x[0, :, :, :].detach().cpu())
    fig.add_subplot(1, 10, i+1)
    plt.imshow(x)
plt.show()


# In[Save images]

img=[]
for i in range(2): 
    img.append(model.decoder(torch.randn(100).cuda()))
    torchvision.utils.save_image(img[i], os.path.join(os.getcwd(),"to","sample_directory","samples","vae_"+str(i)+".png"),normalize=True)


