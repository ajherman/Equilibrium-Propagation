#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import re

import torch

import pandas as pd

import numpy as np


# In[3]:


import torchvision
import torch.nn.functional as F


# In[4]:


import sys
sys.path.append('../')
from output_utils import *


# In[5]:


pd.options.display.float_format = '{:,.2f}'.format


# In[6]:


runs = getruns('../Laborieux-Arch/results/EP/*/*/*/')


# In[7]:


runsdf = pd.DataFrame(runs)
runsdf.sort_values('test', ascending=False)[['task', 'model', 'test', 'train', 'softmax', 'loss', 'load_path', 'epochs', 'load_path_convert', 'competitiontype', 'inhibitstrength', 'pools', 'lat_constraints', 'dir']]


# In[8]:


path = '../Laborieux-Arch/results/EP/mse/2023-07-24/23-13-24_gpu0_23epochs/'
lattest = runsdf[runsdf['dir'] == path]
latmodel = torch.load(path + 'model.pt')
lattest


# In[9]:


path = '../Laborieux-Arch/results/EP/cel/2023-07-24/origcodebase_09-30-41_gpu0_15epochs/'
controltest = runsdf[runsdf['dir'] == path]
controlmodel = torch.load(path + 'model.pt')
controltest


# In[10]:


mbs = lattest.mbs.item()
T1 = lattest.T1.item()
device = torch.device(lattest.device.item())


# In[11]:


if controltest.task.item()=='MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=mbs, shuffle=True, num_workers=0)

    mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=200, shuffle=False, num_workers=0)

elif controltest.task.item()=='CIFAR10':
    if controltest.data_aug.item():
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])
    else:
         transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                      std=(3*0.2023, 3*0.1994, 3*0.2010)) ])

    cifar10_train_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=True, transform=transform_train, download=True)
    cifar10_test_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=False, transform=transform_test, download=True)

    # For Validation set
    val_index = np.random.randint(10)
    val_samples = list(range( 5000 * val_index, 5000 * (val_index + 1) ))

    #train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, sampler = torch.utils.data.SubsetRandomSampler(val_samples), shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=200, shuffle=False, num_workers=1)


# In[12]:


def evaluate(model, loader, T, device):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T) # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        correct += (y == pred).sum().item()


    acc = correct/len(loader.dataset)
    print(phase+' accuracy :\t', acc)
    return correct


# In[13]:


# PGD attack
# histogram of deviations


# In[14]:


def showattacks(attackx, x, attackpreds, origpreds):
    fig, axs = plt.subplots(1, len(attackx), figsize=(10,1))
    plt.ylabel("original")
    for idx in range(len(attackx)):
        axs[idx].imshow(x[idx][0][0].data.cpu())
        axs[idx].set_title("pred:" + str(origpreds[idx].max(1).indices[0].data.item()))

    fig, axs = plt.subplots(1, len(attackx), figsize=(10,1))
    plt.ylabel("attacked")
    for idx in range(len(attackx)):
        axs[idx].imshow(attackx[idx][0][0].data.cpu())
        axs[idx].set_title("pred:" + str(attackpreds[idx].max(1).indices[0].data.item()))

    fig, axs = plt.subplots(1, len(attackx), figsize=(10,1))
    plt.ylabel("diff")
    for idx in range(len(attackx)):
        axs[idx].imshow(attackx[idx][0][0].data.cpu() - x[idx][0][0].data.cpu())
        axs[idx].set_title("pred:" + str(attackpreds[idx].max(1).indices[0].data.item()))


# In[23]:



from EP_attacks.art_PGD_EP import ProjectedGradientDescentPyTorch
from EP_attacks.pytorch import PyTorchClassifier


# In[19]:


args = lattest
if args.loss.item()=='mse':
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
elif args.loss.item()=='cel':
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)


# In[24]:
not_mse = criterion.__class__.__name__.find('MSE')==-1

class energyModelWrapper(torch.nn.Module):
    def __init__(self):
        super(energyModelWrapper, self).__init__()
    def setup(self, model, y, neurons, beta, T):
        self.model = model
        self.y = y
        self.neurons = neurons
        self.beta = beta
        self.T = 5
    def forward(self, x):
        #xnograd = x.detach()
        #x.retain_grad()
        #self.neurons.retain_graph()
        #self.neurons = self.model.forward(xnograd, self.y, self.neurons, beta=self.beta,
        #                                  T=5, check_thm=False)
        #self.neurons = self.model.forward(x, self.y, self.neurons, beta=self.beta,
        #                                  T=2, check_thm=True)
        self.neurons = self.model.init_neurons(mbs, device)

        neurons = self.neurons
        for t in range(self.T):
            phi = self.model.Phi(x, y, neurons, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            if self.T < 30:
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True, retain_graph=True)
            else:
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

            for idx in range(len(neurons)-1):
                neurons[idx] = self.model.activation( grads[idx] )
                if self.T > 29:
                    neurons[idx].requires_grad = True

            if not_mse and not(self.model.softmax):
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.model.activation( grads[-1] )

            if self.T > 29:
                neurons[-1].requires_grad = True
        self.model.zero_grad()
        if not self.model.softmax:
            return self.neurons[-1]
        else:
            return self.model.synapses[-1](self.neurons[-1])
    def zero_grad(self):
        super(energyModelWrapper, self).zero_grad()
        self.model.zero_grad()


# In[15]:


device = torch.device(0)
latmodel = latmodel.to(device)
controlmodel = controlmodel.to(device)

# In[26]:


x, y = next(iter(train_loader))
x = x.to(device)
y = y.to(device)
neurons = latmodel.init_neurons(mbs, device)
beta = 0.0
T = 250
eps = 0.1
fakemodel = energyModelWrapper()
fakemodel.setup(latmodel, y, neurons, beta, T)

lat_art = PyTorchClassifier(fakemodel, loss=criterion, nb_classes=latmodel.nc, input_shape=(x.shape, y.shape))
latPGD = ProjectedGradientDescentPyTorch(lat_art, 2, eps, 2.5*eps/20, max_iter=20, batch_size=mbs)
attack = latPGD.generate(x.cpu().numpy(), y.cpu().numpy(), retain_graph=True)


print('attack')

matplotlib.use('TkAgg')
plt.figure()
plt.imshow(attack[0].transpose())
plt.show()
plt.savefig('attack1lat.png')
