__author__ = 'Cedric'

import numpy as np
from scipy import stats, integrate
from torch import nn
import torch
from torch.autograd import Variable
import samplers
import math
import matplotlib.pyplot as plt


class JSDLoss(nn.Module):
    """Jensen Shannon Loss"""
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, real, fake):
        return torch.tensor(math.log(2)) + (torch.log(real)).mean()/2. + (torch.log(1-fake)).mean()/2.


class JSDLoss2(nn.Module):
    """Jensen Shannon Loss v2"""
    def __init__(self):
        super(JSDLoss2, self).__init__()

    def forward(self, f_0_estim, f_1_estim):
        return (torch.log(f_1_estim)).mean() + (torch.log(1-f_0_estim)).mean()


class WDLoss(nn.Module):
    """Jensen Shannon Loss v2"""
    def __init__(self):
        super(WDLoss, self).__init__()

    def forward(self, real, fake, gradient_z, lambda_gp):
        return real.mean() - fake.mean() - lambda_gp * ((gradient_z.norm(p=2, dim=1) - 1) ** 2).mean()



class Discriminator(nn.Module):
    """MLP Discriminator"""
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
            )


    def forward(self, x):
        return self.mlp_layers(x)


class Critic(nn.Module):
    """MLP Discriminator"""
    def __init__(self, input_size):
        super(Critic, self).__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,1),
        
            )


    def forward(self, x):
        return self.mlp_layers(x)



if __name__ == '__main__':

    batch_size = 512

    phi_values = np.arange(-1., 1.1, 0.1)
    print(phi_values)

    best_list = []

    # Problem 1.3 - JSD
    for phi in phi_values:

        d = Discriminator(input_size=2)
        optimizer = torch.optim.SGD(d.parameters(), lr=1e-1)
        criterion = JSDLoss()
        real_dist = iter(samplers.distribution1(x=0, batch_size=batch_size))
        fake_dist = iter(samplers.distribution1(x=phi, batch_size=batch_size))
        best_loss = -float('Inf')
        
        for batch in range(2500):
            real_samples = torch.as_tensor(next(real_dist), dtype=torch.float32).view(-1, 2)
            fake_samples = torch.as_tensor(next(fake_dist), dtype=torch.float32).view(-1, 2)
            optimizer.zero_grad()
            real_outputs = d(real_samples)
            fake_outputs = d(fake_samples)
            loss = -criterion(real_outputs, fake_outputs)
            print(loss)
            loss.backward()
            optimizer.step()
            if -loss > best_loss:
                best_loss = -loss
            print(str(batch)+" "+str(best_loss))

        best_list.append(best_loss)

    plt.plot(phi_values, best_list, 's')
    plt.xlabel('$\phi$')
    plt.ylabel('JSD')
    plt.title('Jensen-Shannon Divergence per values of $\phi$')
    plt.savefig("problem1-3-JSD.png")
    plt.show()


    # Problem 1.3 - WD

    for phi in phi_values:
        t = Critic(input_size=2)
        
        optimizer = torch.optim.SGD(t.parameters(), lr=1e-3)
        criterion = WDLoss()
        real_dist = iter(samplers.distribution1(x=0, batch_size=batch_size))
        fake_dist = iter(samplers.distribution1(x=phi, batch_size=batch_size))
        unif = iter(samplers.uniform(batch_size=batch_size))
        best_loss = -float('Inf')
        
        for batch in range(10000):
            real_samples = torch.as_tensor(next(real_dist), dtype=torch.float32).view(-1, 2)
            fake_samples = torch.as_tensor(next(fake_dist), dtype=torch.float32).view(-1, 2)
            a = torch.as_tensor(next(unif), dtype=torch.float32).view(-1, 1)
            z = Variable(a * real_samples + (1 - a) * fake_samples, requires_grad=True)
            optimizer.zero_grad()
            real_outputs = t(real_samples)
            fake_outputs = t(fake_samples)
            z_out = t(z)
            gradient_z = torch.autograd.grad(outputs = z_out, inputs = z, grad_outputs=torch.ones(z_out.size()),
                                             create_graph=True, retain_graph=True, only_inputs=True)[0]
            loss = -criterion(real_outputs, fake_outputs, gradient_z, 100)
            loss.backward()
            optimizer.step()
            if -loss > best_loss:
                best_loss = -loss
            print(str(batch)+" "+str(-loss))


        best_list.append(best_loss)

    plt.plot(phi_values, best_list, "o")
    plt.xlabel('$\phi$')
    plt.ylabel('WD')
    plt.title('Wasserstein distance per values of $\phi$')
    plt.savefig("problem1-3-WD.png")
    plt.show()



    # Problem 1.4
    d_4 = Discriminator(input_size=1)

    optimizer = torch.optim.SGD(d_4.parameters(), lr=1e-3)
    criterion = JSDLoss2()
    f_0 = iter(samplers.distribution3(batch_size=batch_size))
    f_1 = iter(samplers.distribution4(batch_size=batch_size))
    best_loss = -float('Inf')

    for batch in range(10000):
        f_0_samples = torch.as_tensor(next(f_0), dtype=torch.float32)
        f_1_samples = torch.as_tensor(next(f_1), dtype=torch.float32)

        optimizer.zero_grad()

        f_0_outputs = d_4(f_0_samples)
        f_1_outputs = d_4(f_1_samples)
    
        print((torch.log(1-f_0_outputs)).mean())

        loss = -criterion(f_0_outputs, f_1_outputs)
        loss.backward()

        optimizer.step()

        if -loss > best_loss:
            best_loss = -loss
            torch.save(d_4.state_dict(), './best_params.pt')
        print(str(batch)+" "+str(best_loss))