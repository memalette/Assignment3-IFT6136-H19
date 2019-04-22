# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:48:19 2019

@author: Remi
"""

import torch
import torch.optim as optim
import torch.utils.data
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from scipy.special import logsumexp
import numpy as np
import os
from import_data import obtain, load_mnist
from VAEs import VAE_q2



# In[Load and importantion code]

# Import binarized MNIST if it was not already done
exists = os.path.isfile(os.path.join(os.getcwd(),'binarized_mnist'))
if os.path.isdir(os.path.join(os.getcwd(),'binarized_mnist'))==False:
    obtain(os.path.join(os.getcwd(),'binarized_mnist')) 


batch_size=32    
valid_batch_size=200
test_batch_size=200
train_MNISt,val_MNIST,test_MNIST=load_mnist(batch_size,valid_batch_size,test_batch_size)




# In[Train function]
   

def train(epochs, l_r):
    model = VAE_q2() 
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
        
        for batch_idx, (inputs, _ ) in enumerate(train_MNISt):
            if cuda_available:
                inputs = inputs.cuda()
            
            optimizer.zero_grad()
            recon_x, mus, log_vars = model(inputs)
            
            loss, recon_loss, D_KL = model.loss(inputs, recon_x, mus, log_vars)
    
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
            #if batch_idx%50==0:
            print('Epoch : %d Train Loss: %d Recon Loss: %d DKL Loss: %d Prev Loss_train %d Prev Loss_valid %d Progress : %.3f ' % (epoch, loss.item(),recon_loss/inputs.shape[0], D_KL/inputs.shape[0], prev_loss_train,prev_loss_valid, batch_idx/len(train_MNISt)))
        
        loss_train=loss_train/len(train_MNISt)
        losses_train.append(loss_train)
        # Evaluate
        
        model.eval()
        
        best_val_so_far=1000
        for batch_idx, (inputs, _) in enumerate(val_MNIST):
            if cuda_available:
                inputs = inputs.cuda()
    
            recon_x, mus, log_vars = model(inputs)
            loss, recon_loss, D_KL = model.loss(inputs, recon_x, mus, log_vars)
            loss_valid += loss.item()
            
        loss_valid = loss_valid/len(val_MNIST)
        print('Epoch : %d Valid Loss : %.3f ' % (epoch, loss_valid))
        losses_valid.append(loss_valid)
        
        #Saving model if best
        if loss_valid < best_val_so_far:
            best_val_so_far = loss_valid
            print("Saving model parameters to best_params.pt")
            torch.save(model.state_dict(), os.path.join(os.getcwd(),'best_model_vae_q2', 'best_params_0417.pt'))

        model.train()
        
        return losses_train, losses_valid


# In[Train model]
        
losses_train, losses_valid = train(25, 3e-4)

           
# In[]
model=VAE_q2()

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

# Where the best model is saved
path=os.path.join(os.getcwd(),'best_model_vae_q2', 'best_params_0417.pt')

# load the best model
model.load_state_dict(torch.load(path)) 
  
    
 
# In[Likelihood function]   

 
def calculate_likelihood(X, S):
    
    model.eval()
    
    
    N_obs  = X.size(0)
    N_sim  = S.size(1)
    S      = torch.transpose(S,1,0)    
        
    
    # parameters from the encoder 
    z_mu, z_logvar = model.encoder(X)
    z_mu=z_mu.detach()
    
    z_logvar=z_logvar.detach()
    
    std = z_logvar.mul(0.5).exp_()
    std=std.detach()
    #We simulate the hidden state for the K number of simulations
    z = z_mu+ std * S
    z=z.detach()
    
    RE=torch.zeros((N_sim,N_obs)).cuda()
    
    
    for j in range(N_sim):
        ber = Bernoulli(model.decoder(z[j,:,:]).detach())
        RE[j] = ber.log_prob(X).sum(1).detach()
    RE=RE.detach()
    log_norm = MultivariateNormal(torch.zeros(100).cuda(),torch.diag(torch.ones(100)).cuda())
    
    log_p_z = log_norm.log_prob(z)
    log_p_z=log_p_z.detach()
    log_mult = Normal(z_mu,std)
 
    log_q_z = log_mult.log_prob(z).sum(2)
    log_q_z=log_q_z.detach()
    KL = -(log_p_z - log_q_z)
    
    L = (RE - KL)
    
    log_lik = logsumexp( L.detach().cpu().numpy() , axis=0)
    
    log_lik=(log_lik - np.log(N_sim))

    return log_lik


# In[Calculate the likelihood]
batch_size=32    
valid_batch_size=1000
test_batch_size=1000
_ ,val_MNIST,test_MNIST=load_mnist(batch_size,valid_batch_size,test_batch_size)


model.eval()
log_lik_val = []
for batch_idx, (X, _) in enumerate(val_MNIST):
    S = torch.randn(X.shape[0],200,100).cuda()
    if cuda_available:
        X = X.cuda()
        S = S.cuda()
    log_lik_val.append(calculate_likelihood(X, S))
    print(batch_idx)
log_lik_val = np.array(log_lik_val)
np.mean(log_lik_val)    


log_lik_test = []
for batch_idx, (X, _) in enumerate(test_MNIST):
    S = torch.randn(X.shape[0],200,100).cuda()
    if cuda_available:
        X = X.cuda()
        S = S.cuda()
    log_lik_test.append(calculate_likelihood(X, S))
    print(batch_idx)
log_lik_test = np.array(log_lik_test)
np.mean(log_lik_test)    
