import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
from data_utils import *

from EP_attacks.art_PGD_EP import ProjectedGradientDescentPyTorch
from EP_attacks.pytorch import PyTorchClassifier

from itertools import repeat
from torch.nn.parameter import Parameter
import collections
import matplotlib
matplotlib.use('Agg')


# Activation functions
def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(2*x-1))*0.5

def ctrd_hard_sig(x):
    return (F.hardtanh(2*x))*0.5

def my_hard_sig(x):
    return (1+F.hardtanh(x-1))*0.5





# Some helper functions
def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_(n.requires_grad))
    return copy






def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxPool2d(2, stride=2) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( torch.nn.Identity() )
        elif letters[p]=='M':
            pools.append( torch.nn.MaxPool2d(4, stride=4) )
    return pools
        


       
def my_init(scale):
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            # add identity to prevent arbitrary geodesics from scrambling seperability, losing information
            n = min(m.weight.size(0), m.weight.size(1))
            centerx = math.floor(m.kernel_size[0]/2)
            centery = math.floor(m.kernel_size[1]/2)
            with torch.no_grad():
                m.weight[:n,:n,centerx,centery].add_(torch.eye(n, device=m.weight.device))
            ##### """
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                #m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            # """
            n = min(m.weight.size(0), m.weight.size(1))
            with torch.no_grad():
                m.weight[:n,:n].add_(torch.eye(n, device=m.weight.device))
            ##### """
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                #m.bias.data.mul_(scale)
    return my_scaled_init


        
# Multi-Layer Perceptron

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(P_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False    #Softmax readout is only defined for CNN and VFCNN       
        self.nc = self.archi[-1]

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

            
    def Phi(self, x, y, neurons, beta, criterion):
        #Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss 
        x = x.view(x.size(0),-1) # flattening the input
        
        layers = [x] + neurons  # concatenate the input to other layers
        
        # Primitive function computation
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.W.s_n-1
        
        if beta!=0.0: # Nudging the output layer when beta is non zero 
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi -= beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) # dPhi/ds

            for idx in range(len(neurons)-1):
                neurons[idx] = self.activation( grads[idx] )  # s_(t+1) = sigma( dPhi/ds )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            if not_mse:
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation( grads[-1] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
        

# MLP with analytically computed energy model rule (rather than computing grad of Phi)
class MLP_Analytical(P_MLP):
    def __init__(self, archi, activation=torch.tanh):
        super(MLP_Analytical, self).__init__(archi, activation)

    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
        x = x.view(x.size(0),-1) # flattening the input
        layers = [x] + neurons 
        dn = [] # tendency of neurons
        for idx in range(1, len(layers)): # exclude input layer
            dn.append(torch.zeros_like(layers[idx]))

        with torch.no_grad():
            for t in range(T):
                for idx in range(1, len(layers)-1):
                    dn[idx-1] = self.synapses[idx-1](layers[idx-1]) # input from previous layer
                    dn[idx-1] += torch.matmul(self.synapses[idx].weight.t(), (layers[idx+1]).T ).T # input from next layer ## layers[idx+1]-self.synapses[idx].bias
                nudge = beta * (F.one_hot(y, num_classes=self.nc) - layers[-1]) # is grad(MSE(y, pred)) w/r to neurons
                dn[-1] = self.synapses[-1](layers[-2]) + nudge 
                for idx in range(len(dn)):
                    dn[idx] -= layers[idx+1] # exponential decay 
                    layers[idx+1] = self.activation(layers[idx+1] + dn[idx])
        
        return layers[1:]
            
    
         
# Vector Field Multi-Layer Perceptron

class VF_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(VF_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False        
        self.nc = self.archi[-1]

        # Forward synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

        # Backward synapses
        self.B_syn = torch.nn.ModuleList()
        for idx in range(1, len(archi)-1):            
            self.B_syn.append(torch.nn.Linear(archi[idx+1], archi[idx], bias=False))


    def Phi(self, x, y, neurons, beta, criterion, neurons_2=None):
        # For assymetric connections each layer has its own phi 
        x = x.view(x.size(0),-1)
        
        layers = [x] + neurons
        
        phis = []

        if neurons_2 is None:  # Standard case for the dynamics
            for idx in range(len(self.synapses)-1):
                phi = torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze()
                phi += torch.sum( self.B_syn[idx](layers[idx+2]) * layers[idx+1], dim=1).squeeze()
                phis.append(phi)        

            phi = torch.sum( self.synapses[-1](layers[-2]) * layers[-1], dim=1).squeeze()
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()     
                phi -= beta*L
        
            phis.append(phi)
        
        else:  # Used only for computing the vector field EP update
            layers_2 = [x] + neurons_2
            for idx in range(len(self.synapses)-1):
                phi = torch.sum( self.synapses[idx](layers[idx]) * layers_2[idx+1], dim=1).squeeze()
                phi += torch.sum( self.B_syn[idx](layers[idx+2]) * layers_2[idx+1], dim=1).squeeze()
                phis.append(phi)        

            phi = torch.sum( self.synapses[-1](layers[-2]) * layers_2[-1], dim=1).squeeze()
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers_2[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers_2[-1].float(), y).squeeze()     
                phi -= beta*L
        
            phis.append(phi)

        return phis
    
    
    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
        
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phis = self.Phi(x, y, neurons, beta, criterion)
            for idx in range(len(neurons)-1):
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=check_thm)

                neurons[idx] = self.activation( grad[0] )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=check_thm)
            if not_mse:
                neurons[-1] = grad[0]
            else:
                neurons[-1] = self.activation( grad[0] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_2)
     
        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
            delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
        

   
    
    
# Convolutional Neural Network

class P_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False):
        # super(P_CNN, self).__init__()
        if not hasattr(self, 'call_super_init') or self.call_super_init:
            super(P_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]        

        self.activation = activation
        self.pools = pools
        
        self.synapses = torch.nn.ModuleList()
        
        self.softmax = softmax # whether to use softmax readout or not

        size = in_size # size of the input : 32 for cifar10

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], padding=paddings[idx], bias=True))
                
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        


    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len):    
                phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
            for idx in range(conv_len, tot_len):
                phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi -= beta*L

        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len):
                phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
            for idx in range(conv_len, tot_len-1):
                phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()  
                phi -= beta*L            
        return phi

    def dPhi(self, grads, x, y, neurons, beta, criterion):
        #unpoolidxs = [None for range(len(self.pools))]
        for p in self.pools:
            p.return_indices = True
        grads[0] += self.pools[0](self.synapses[0](x))[0]
        for idx in range(1,len(self.kernels)):
            out = self.pools[idx](self.synapses[idx](neurons[idx-1]))
            grads[idx] += out[0]
            
            grads[idx-1] += F.conv_transpose2d(F.max_unpool2d(neurons[idx], out[1], self.pools[idx-1].kernel_size), self.synapses[idx].weight, stride=self.synapses[idx].stride, padding=self.synapses[idx].padding)
        
        tot_len = len(self.synapses)
        if self.softmax:
            tot_len -= 1
        for idx in range(len(self.kernels), tot_len):
            grads[idx] += self.synapses[idx](neurons[idx-1].view(x.size(0), -1))
            grads[idx-1] += (F.linear(neurons[idx], self.synapses[idx].weight.T)).view(grads[idx-1].size())

        if self.softmax:
            print('Analytical CEL Softmax not implemented!')
            pass # idk the grad of softmax
        elif criterion.__class__.__name__.find('MSE') != -1:
            y = F.one_hot(y, num_classes=self.nc)
            grads[-1] += beta*(y - neurons[-1])
        for p in self.pools:
            p.return_indices = False

        return grads

    

    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        if check_thm:
            for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].retain_grad()
             
                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].retain_grad()
        else:
             for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].requires_grad = True
             
                
                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                    #neurons[-1] = self.activation(grads[-1]) + 1e-2
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].requires_grad = True

        return neurons
       

    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
 
           
   


# CNN with analytically computed energy model rule (equation of motion rather than computing grad of Phi computationaly)
class CNN_Analytical(P_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False):
        super(CNN_Analytical, self).__init__(in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False)


        # define transposes of each layer to send spikes backwards
        self.syn_transpose = torch.nn.ModuleList()
        for idx in range(len(self.synapses)):
            layer = self.synapses[idx]
            if isinstance(layer, torch.nn.Conv2d):
                transpose = torch.nn.ConvTranspose2d(layer.out_channels, layer.in_channels, layer.kernel_size, stride=layer.stride,
                                        padding=layer.padding, dilation=layer.dilation, bias=False)
                transpose.weight.data = layer.weight.data
                # transpose.bias.data = torch.zeros_like(transpose.bias.data)
            elif isinstance(layer, torch.nn.Linear):
                transpose = torch.nn.Linear(layer.out_features, layer.in_features, bias=False)
                transpose.weight.data = layer.weight.data.T
            self.syn_transpose.append(transpose)
        self.unpools = []
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = True # we will need the index information to unpool
            pool = self.pools[idx]
            if isinstance(pool, torch.nn.MaxPool2d):
                self.unpools.append(torch.nn.MaxUnpool2d(pool.kernel_size, stride=pool.stride)) 
            elif isinstance(pool, torch.nn.AvgPool2d):
                avgunpool = torch.nn.ConvTranspose2d(1, 1, pool.kernel_size, stride=pool.stride, padding=0, dilation=0, bias=False)
                avgunpool.weight.fill_(1/pool.kernel_size**2)
                self.unpools.append(avgunpool)

    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
        mbs = x.size(0)
        layers = [x] + neurons 
        dn = [] # tendency of neurons
        poolidxs = [[] for i in range(len(self.pools))]
        for n in neurons: # exclude input layer
            dn.append(torch.zeros_like(n, device=n.device))

        with torch.no_grad():
            for t in range(T):
                # forwards connections (input from previous layer)
                for idx in range(0, len(self.kernels)):
                    dn[idx], poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx]))
                for idx in range(len(self.kernels), len(self.synapses)):
                    dn[idx] = self.synapses[idx](layers[idx].view(mbs,-1))
                # print([torch.isnan(dni).any() for dni in dn])
                # print('^^^^^ post forwards')
                
                # backwards connections (input from following layer)
                for idx in range(0, len(self.kernels)-1):
                    dn[idx] += self.syn_transpose[idx+1](self.unpools[idx+1](layers[idx+2], poolidxs[idx+1]))
                for idx in range(len(self.kernels)-1, len(self.synapses)-1):
                    dn[idx] += self.syn_transpose[idx+1](layers[idx+2]).view(dn[idx].size())
                # nudge and final layer
                if criterion.__class__.__name__.find('MSE')!=-1:
                    nudge = beta * 2 * (F.one_hot(y, num_classes=self.nc) - layers[-1]) # is grad(MSE(y, pred)) w/r to neurons
                elif criterion.__class__.__name__.find('CrossEntropy')!=-1:
                    nudge = beta * F.one_hot(y, num_classes=self.nc) / (layers[-1] + 1e-3) # gradient of cross entropy y*log(yhat)
                dn[-1] += nudge # final layer only has input from previous and nudge
                # update neurons by tendencies
                for idx in range(len(dn)):
                    dn[idx] -= layers[idx+1] # exponential decay 
                    layers[idx+1] = self.activation(layers[idx+1] + dn[idx])
        
        return layers[1:] # neurons (layers excluding input)

    def postupdate(self):
        for idx in range(len(self.synapses)):
            layer = self.synapses[idx]
            transpose = self.syn_transpose[idx]
            if isinstance(layer, torch.nn.Conv2d):
                transpose.weight.data = layer.weight.data
                # transpose.bias.data = torch.zeros_like(tranpose.bias.data)
            elif isinstance(layer, torch.nn.Linear):
                transpose.weight.data = layer.weight.data.T

        

 
# Vector Field Convolutional Neural Network

class VF_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax = False, same_update=False):
        super(VF_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]        

        self.activation = activation
        self.pools = pools
        
        self.synapses = torch.nn.ModuleList()  # forward connections
        self.B_syn = torch.nn.ModuleList()     # backward connections

        self.same_update = same_update         # whether to use the same update for forward ans backward connections
        self.softmax = softmax                 # whether to use softmax readout


        size = in_size

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], padding=paddings[idx], bias=True))
                
            if idx>0:  # backward synapses except for first layer
                self.B_syn.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx],
                                                      stride=strides[idx], padding=paddings[idx], bias=False))

            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]
        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
            if not(self.softmax and (idx==(len(fc)-1))):
                self.B_syn.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=False))


    def angle(self): # computes the angle between forward and backward weights
        angles = []
        with torch.no_grad():
            for idx in range(len(self.B_syn)):
                fnorm = self.synapses[idx+1].weight.data.pow(2).sum().pow(0.5).item()
                bnorm = self.B_syn[idx].weight.data.pow(2).sum().pow(0.5).item()
                cos = self.B_syn[idx].weight.data.mul(self.synapses[idx+1].weight.data).sum().div(fnorm*bnorm)
                angle = torch.acos(cos).item()*(180/(math.pi))
                angles.append(angle)                
        return angles


    def Phi(self, x, y, neurons, beta, criterion, neurons_2=None):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)
        bck_len = len(self.B_syn)

        layers = [x] + neurons        
        phis = []

        if neurons_2 is None: #neurons 2 is not None only when computing the EP update for different updates between forward and backward
            #Phi computation changes depending on softmax == True or not
            if not self.softmax:

                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)

                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers[conv_len], dim=(1,2,3)).squeeze()
                phi += torch.sum( self.B_syn[conv_len-1](layers[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                phis.append(phi)            

                for idx in range(conv_len+1, tot_len-1):
                    phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                    phi += torch.sum( self.B_syn[idx](layers[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                    phis.append(phi)

                phi = torch.sum( self.synapses[-1](layers[-2].view(mbs,-1)) * layers[-1], dim=1).squeeze()
                if beta!=0.0:
                    if criterion.__class__.__name__.find('MSE')!=-1:
                        y = F.one_hot(y, num_classes=self.nc)
                        L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                    else:
                        L = criterion(layers[-1].float(), y).squeeze()             
                    phi -= beta*L
                phis.append(phi)

            else:
                #the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)
            
                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers[conv_len], dim=(1,2,3)).squeeze()
                if bck_len>=conv_len:
                    phi += torch.sum( self.B_syn[conv_len-1](layers[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                    phis.append(phi)            

                    for idx in range(conv_len, tot_len-2):
                        phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                        phi += torch.sum( self.B_syn[idx](layers[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                        phis.append(phi)
                    
                    phi = torch.sum( self.synapses[-2](layers[-2].view(mbs,-1)) * layers[-1], dim=1).squeeze()
                    if beta!=0.0:
                        L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                        phi -= beta*L
                    phis.append(phi)                    

                #the prediction is made with softmax[last weights[penultimate layer]]
                elif beta!=0.0:
                    L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                    phi -= beta*L
                    phis.append(phi)            
                else:
                    phis.append(phi)
           
        else:
            layers_2 = [x] + neurons_2
            #Phi computation changes depending on softmax == True or not
            if not self.softmax:

                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers_2[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers_2[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)

                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers_2[conv_len], dim=(1,2,3)).squeeze()
                phi += torch.sum( self.B_syn[conv_len-1](layers_2[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                phis.append(phi)            

                for idx in range(conv_len+1, tot_len-1):
                    phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers_2[idx+1], dim=1).squeeze()
                    phi += torch.sum( self.B_syn[idx](layers_2[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                    phis.append(phi)

                phi = torch.sum( self.synapses[-1](layers[-2].view(mbs,-1)) * layers_2[-1], dim=1).squeeze()
                if beta!=0.0:
                    if criterion.__class__.__name__.find('MSE')!=-1:
                        y = F.one_hot(y, num_classes=self.nc)
                        L = 0.5*criterion(layers_2[-1].float(), y.float()).sum(dim=1).squeeze()   
                    else:
                        L = criterion(layers_2[-1].float(), y).squeeze()             
                    phi -= beta*L
                phis.append(phi)

            else:
                #the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers_2[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers_2[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)
            
                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers_2[conv_len], dim=(1,2,3)).squeeze()
                if bck_len>=conv_len:
                    phi += torch.sum( self.B_syn[conv_len-1](layers_2[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze() 
                    phis.append(phi)            

                    for idx in range(conv_len, tot_len-2):
                        phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers_2[idx+1], dim=1).squeeze()
                        phi += torch.sum( self.B_syn[idx](layers_2[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                        phis.append(phi)
                     
                    phi = torch.sum( self.synapses[-2](layers[-2].view(mbs,-1)) * layers_2[-1], dim=1).squeeze()
                    if beta!=0.0:
                        L = criterion(self.synapses[-1](layers_2[-1].view(mbs,-1)).float(), y).squeeze()             
                        phi -= beta*L
                    phis.append(phi)                    

                #the prediction is made with softmax[last weights[penultimate layer]]
                elif beta!=0.0:
                    L = criterion(self.synapses[-1](layers_2[-1].view(mbs,-1)).float(), y).squeeze()             
                    phi -= beta*L                       
                    phis.append(phi)
                else:
                    phis.append(phi)
        return phis
    

    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        if check_thm:
            for t in range(T):
                phis = self.Phi(x, y, neurons, beta, criterion)
                for idx in range(len(neurons)-1):
                    init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                    grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=True)

                    neurons[idx] = self.activation( grad[0] )
                    neurons[idx].retain_grad()
             
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=True)
                if not_mse and not(self.softmax):
                    neurons[-1] = grad[0]
                else:
                    neurons[-1] = self.activation( grad[0] )

                neurons[-1].retain_grad()
        else:
             for t in range(T):
                phis = self.Phi(x, y, neurons, beta, criterion)
                for idx in range(len(neurons)-1):
                    init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                    grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=False)

                    neurons[idx] = self.activation( grad[0] )
                    neurons[idx].requires_grad = True
             
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=False)
                grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=False)
                if not_mse and not(self.softmax):
                    neurons[-1] = grad[0]
                else:
                    neurons[-1] = self.activation( grad[0] )

                neurons[-1].requires_grad = True

        return neurons
       

    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )  # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we remove the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False, neurons_3=None):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            if neurons_3 is None: # neurons_3 is not None only when doing thirdphase with old VF (same update False) 
                phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)   # phi will habe the form s_* W s_*
            else:
                phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion, neurons_2=neurons_2) # phi will have the form s_* W s_beta
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_2, criterion)

        if self.same_update:
            phis_2 = self.Phi(x, y, neurons_2, beta_2, criterion)  # Phi = s_beta W s_beta
        else:
            if neurons_3 is None:
                phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_2) # phi = s_* W s_beta
            else:
                phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_3) # phi = s_* W s_-beta
                
        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = ((phi_2 - phi_1)/(beta_1 - beta_2))        
            delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
        if self.same_update:
            with torch.no_grad():
                for idx in range(len(self.B_syn)):
                    common_update = 0.5*(self.B_syn[idx].weight.grad.data + self.synapses[idx+1].weight.grad.data)
                    self.B_syn[idx].weight.grad.data.copy_(common_update)
                    self.synapses[idx+1].weight.grad.data.copy_(common_update)
       

         

def check_gdu(model, x, y, T1, T2, betas, criterion, alg='EP'):
    # This function returns EP gradients and BPTT gradients for one training iteration
    #  given some labelled data (x, y), time steps for both phases and the loss
    
    # Initialize dictionnaries that will contain BPTT gradients and EP updates
    BPTT, EP = {}, {}
    if alg=='CEP':
        prev_p = {}

    for name, p in model.named_parameters():
        BPTT[name], EP[name] = [], []
        if alg=='CEP':
            prev_p[name] = p

    neurons = model.init_neurons(x.size(0), x.device)
    for idx in range(len(neurons)):
        BPTT['neurons_'+str(idx)], EP['neurons_'+str(idx)] = [], []
    
    # We first compute BPTT gradients
    # First phase up to T1-T2
    beta_1, beta_2 = betas
    neurons = model(x, y, neurons, T1-T2, beta=beta_1, criterion=criterion)
    ref_neurons = copy(neurons)
    
    
    # Last steps of the first phase
    for K in range(T2+1):

        neurons = model(x, y, neurons, K, beta=beta_1, criterion=criterion) # Running K time step 

        # detach data and neurons from the graph
        x = x.detach()
        x.requires_grad = True
        leaf_neurons = []
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
            leaf_neurons.append(neurons[idx])

        neurons = model(x, y, neurons, T2-K, beta=beta_1, check_thm=True, criterion=criterion) # T2-K time step
        
        # final loss
        if criterion.__class__.__name__.find('MSE')!=-1:
            loss = (1/(2.0*x.size(0)))*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).squeeze()
        else:
            if not model.softmax:
                loss = (1/(x.size(0)))*criterion(neurons[-1].float(), y).squeeze()
            else:
                loss = (1/(x.size(0)))*criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).squeeze()

        # setting gradients field to zero before backward
        neurons_zero_grad(leaf_neurons)
        model.zero_grad()

        # Backpropagation through time
        loss.backward(torch.tensor([1 for i in range(x.size(0))], dtype=torch.float, device=x.device, requires_grad=True))

        # Collecting BPTT gradients : for parameters they are partial sums over T2-K time steps
        if K!=T2:
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append( update.unsqueeze(0) )  # unsqueeze for time dimension
                neurons = copy(ref_neurons) # Resetting the neurons to T1-T2 step
        if K!=0:
            for idx in range(len(leaf_neurons)):
                update = torch.empty_like(leaf_neurons[idx]).copy_(grad_or_zero(leaf_neurons[idx]))
                BPTT['neurons_'+str(idx)].append( update.mul(-x.size(0)).unsqueeze(0) )  # unsqueeze for time dimension

                                
    # Differentiating partial sums to get elementary parameter gradients
    for name, p in model.named_parameters():
        for idx in range(len(BPTT[name]) - 1):
            BPTT[name][idx] = BPTT[name][idx] - BPTT[name][idx+1]
        
    # Reverse the time
    for key in BPTT.keys():
        BPTT[key].reverse()
            
    # Now we compute EP gradients forward in time
    # Second phase done step by step
    for t in range(T2):
        neurons_pre = copy(neurons)                                          # neurons at time step t
        neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)  # neurons at time step t+1
        
        model.compute_syn_grads(x, y, neurons_pre, neurons, betas, criterion, check_thm=True)  # compute the EP parameter update

        if alg=='CEP':
            for p in model.parameters():
                p.data.add_(-1e-5 * p.grad.data)
        
        # Collect the EP updates forward in time
        for n, p in model.named_parameters():
            update = torch.empty_like(p).copy_(grad_or_zero(p))
            EP[n].append( update.unsqueeze(0) )                    # unsqueeze for time dimension
        for idx in range(len(neurons)):
            update = (neurons[idx] - neurons_pre[idx])/(beta_2 - beta_1)
            EP['neurons_'+str(idx)].append( update.unsqueeze(0) )  # unsqueeze for time dimension
        
    # Concatenating with respect to time dimension
    for key in BPTT.keys():
        BPTT[key] = torch.cat(BPTT[key], dim=0).detach()
        EP[key] = torch.cat(EP[key], dim=0).detach()
    
    if alg=='CEP':
        for name, p in model.named_parameters():
            p.data.copy_(prev_p[name])    

    return BPTT, EP
    


def RMSE(BPTT, EP):
    # print the root mean square error, and sign error between EP and BPTT gradients
    print('\nGDU check :')
    for key in BPTT.keys():
        K = BPTT[key].size(0)
        f_g = (EP[key] - BPTT[key]).pow(2).sum(dim=0).div(K).pow(0.5)
        f =  EP[key].pow(2).sum(dim=0).div(K).pow(0.5)
        g = BPTT[key].pow(2).sum(dim=0).div(K).pow(0.5)
        comp = f_g/(1e-10+torch.max(f,g))
        sign = torch.where(EP[key]*BPTT[key] < 0, torch.ones_like(EP[key]), torch.zeros_like(EP[key]))
        print(key.replace('.','_'), '\t RMSE =', round(comp.mean().item(), 4), '\t SIGN err =', round(sign.mean().item(), 4))
    print('\n')


def debug(model, prev_p, optimizer):
    optimizer.zero_grad()
    for (n, p) in model.named_parameters():
        idx = int(n[9]) 
        p.grad.data.copy_((prev_p[n] - p.data)/(optimizer.param_groups[idx]['lr']))
        p.data.copy_(prev_p[n])
    for i in range(len(model.synapses)):
        optimizer.param_groups[i]['lr'] = prev_p['lrs'+str(i)]
        #optimizer.param_groups[i]['weight_decay'] = prev_p['wds'+str(i)]
    optimizer.step()



# helper function to show an image
# from the pytorch documentation
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def showattacks(attackx, x, attackpreds, origpreds, savefig=False, path='/tmp/attack.pdf'):
    fig, axs = plt.subplots(3, len(attackx), figsize=(len(attackx),3))
    axs[0,0].set_ylabel("original")
    for idx in range(len(attackx)):
        axs[0,idx].imshow(x[idx].transpose(1,2,0)/2 + 0.5)
        axs[0,idx].set_title(str(origpreds[idx]))
        axs[0,idx].set_xticks([])
        axs[0,idx].set_yticks([])

    axs[1,0].set_ylabel("attacked")
    for idx in range(len(attackx)):
        axs[1,idx].imshow(attackx[idx].transpose(1,2,0)/2 + 0.5)
        axs[1,idx].set_title(str(attackpreds[idx]))
        axs[1,idx].set_xticks([])
        axs[1,idx].set_yticks([])

    axs[2,0].set_ylabel("diff")
    for idx in range(len(attackx)):
        diff = attackx[idx].transpose(1,2,0) - x[idx].transpose(1,2,0)
        diff -= np.min(diff)
        diff = diff / np.max(diff)
        axs[2,idx].imshow(diff)
        axs[2,idx].set_xticks([])
        axs[2,idx].set_yticks([])
    
    plt.tight_layout()
    if savefig:
        print('saving figure at ', path)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


class energyModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(energyModelWrapper, self).__init__()
        self.model = model
    def setup(self, y, beta, T, criterion):
        self.y = y
        self.beta = beta
        self.T = T
        self.criterion = criterion
    def forward(self, x):
        #xnograd = x.detach()
        #x.retain_grad()
        #self.neurons.retain_graph()
        #self.neurons = self.model.forward(xnograd, self.y, self.neurons, beta=self.beta,
        #                                  T=5, check_thm=False)
        #self.neurons = self.model.forward(x, self.y, self.neurons, beta=self.beta,
        #                                  T=2, check_thm=True)
        mbs = x.size(0)
        device = x.device
        self.neurons = self.model.init_neurons(mbs, device)
      #  not_mse = self.criterion.__class__.__name__.find('MSE')==-1

        self.neurons = self.model.forward(x, self.y, self.neurons, self.T, self.beta, criterion=self.criterion, check_thm=(self.T < 30))

      #  neurons = self.neurons
      #  for t in range(self.T):
      #      phi = self.model.Phi(x, self.y, self.neurons, self.beta, self.criterion)
      #      init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
      #      if self.T < 30:
      #          grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True, retain_graph=True)
      #      else:
      #          grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

      #      for idx in range(len(neurons)-1):
      #          neurons[idx] = self.model.activation( grads[idx] )
      #          if self.T > 29:
      #              neurons[idx].requires_grad = True

      #      if not_mse and not(self.model.softmax):
      #          neurons[-1] = grads[-1]
      #      else:
      #          neurons[-1] = self.model.activation( grads[-1] )

      #      if self.T > 29:
      #          neurons[-1].requires_grad = True
      #  self.model.zero_grad()
        if not self.model.softmax:
            return self.neurons[-1]
        else:
            return F.softmax(self.model.synapses[-1](self.neurons[-1].view(mbs,-1)), dim=1)
    def zero_grad(self):
        super(energyModelWrapper, self).zero_grad()
        self.model.zero_grad()

def attack(model, loader, nbatches, attack_steps, predict_steps, eps, criterion, device, path, norm=2, save_adv_examples=False, figs=False, deltapert=False):
    criterion.reduction='sum'
    mbs = loader.batch_size
    forwardmodel = energyModelWrapper(model)

    #if hasattr(model, 'sparselayeridxs'): # lca models have specialized forward methods
    #    forwardmodel.forward = lambda x: model.forward(x, forwardmodel.y, forwardmodel.neurons, forwardmodel.T, forwardmodel.beta, check_thm=(forwardmodel.T < 30), criterion=forwardmodel.criterion)
    delta_right = list([0 for li in range(len(forwardmodel.model.synapses))])
    delta_wrong = list([0 for li in range(len(forwardmodel.model.synapses))])

    beta = 0.0
    savepath = path + '/adversarial_examples/eps_{}/'.format(eps)
    os.makedirs(savepath, exist_ok=True)
    savepath += '/attack_{}__pred_{}'.format(attack_steps, predict_steps)

    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std = np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    art_model = PyTorchClassifier(forwardmodel, loss=criterion, nb_classes=model.nc, input_shape=(mbs,model.channels[0], model.in_size, model.in_size), preprocessing=(mean,std))
    art_PGD = ProjectedGradientDescentPyTorch(art_model, norm, eps, 2.5*eps/20, max_iter=20, batch_size=mbs, verbose=False)
    originals = []
    adv_examples = []
    preds = []
    preds_adv = []
    
    tot_correct = 0
    tot_correct_adv = 0
    for idx, (x,y) in enumerate(loader):
        if idx >= nbatches:
            break
        # do original, unhampered prediction for control group
        forwardmodel.setup(y, beta, predict_steps, criterion)
        pred = np.argmax(art_model.predict(x.cpu().numpy()), axis=1)
        correct = (torch.from_numpy(pred) == y).sum().item()
        tot_correct += correct

        if deltapert:
            # simulate for T2-T1, so the original prediction has as many steps as the adversarial run
            forwardmodel.neurons = forwardmodel.model.forward(x, forwardmodel.y, forwardmodel.neurons, attack_steps-predict_steps, forwardmodel.beta, criterion=forwardmodel.criterion, check_thm=False)
            
            print('orig L2', [n.norm(2) for n in forwardmodel.neurons])
            neurons_orig = copy(forwardmodel.neurons)

        # design adversarial example and predict with that
        forwardmodel.setup(y, beta, attack_steps, criterion)
        x_adv = (art_PGD.generate(x.cpu().numpy())) # np.vectorize(loader.dataset.transform)
        x_adv.clip(-2, 2) # clip large values (ensures large enough attacks start saturating values)
        forwardmodel.setup(y, beta, predict_steps, criterion)
        pred_adv = np.argmax(art_model.predict(x_adv), axis=1)
        correct_adv = (torch.from_numpy(pred_adv) == y).sum().item()        
        tot_correct_adv += correct_adv

        adv_success = torch.logical_and((torch.from_numpy(pred) == y), (torch.from_numpy(pred_adv) != y))
        print('sh', pred.shape, adv_success.size(), neurons_orig[0].size())
        originals.append(x.cpu().numpy()[adv_success])
        adv_examples.append(x_adv[adv_success])
        pred_name = np.asarray(list(map(lambda i: loader.dataset.classes[i], pred)))
        pred_adv_name = np.asarray(list(map(lambda i: loader.dataset.classes[i], pred_adv)))
        preds.append(pred_name[adv_success])
        preds_adv.append(pred_adv_name[adv_success])

        print('Batch {} of {} : original accuracy={}, adversarial accuracy={}, attack success rate={}'.format(idx, nbatches, correct/y.size(0), correct_adv/y.size(0), adv_success.sum()/y.size(0)))

        if deltapert:
            neurons_adv = copy(forwardmodel.neurons)
            print('adv  L2', [n.norm(2) for n in neurons_adv])

            for li in range(len(neurons_orig)):
                delta_right[li] += ( (neurons_adv[li] - neurons_orig[li])[np.logical_not(adv_success)].norm(2, dim=list(range(1,neurons_orig[li].dim())))/neurons_orig[li][np.logical_not(adv_success)].norm(2,dim=list(range(1,neurons_orig[li].dim()))) ).mean(dim=0) / nbatches
                delta_wrong[li] += ( (neurons_adv[li] - neurons_orig[li])[adv_success].norm(2,dim=list(range(1,neurons_orig[li].dim())))/neurons_orig[li][adv_success].norm(2,dim=list(range(1,neurons_orig[li].dim()))) ).mean(dim=0) / nbatches



        #if (idx/len(loader))%0.1 < 0.001 and adv_success.sum() > 0:
        if figs and adv_success.sum() > 1:
            showattacks(np.asarray(x_adv[adv_success]), np.asarray(x[adv_success]), np.asarray(pred_adv_name[adv_success]), np.asarray(pred_name[adv_success]), savefig=True, path=savepath+'__{}.pdf'.format(idx))

    if save_adv_examples:
        np.save(savepath + '__originals.npy', np.asarray(originals))
        np.save(savepath + '__examples.npy', np.asarray(adv_examples))
        np.save(savepath + '__original_pred.npy', np.asarray(preds))
        np.save(savepath + '__attacked_pred.npy', np.asarray(preds_adv))
#    if deltapert:
#        np.save(savepath + '__delta_pertubation_layers.npy', np.asarray([delta_right, delta_wrong]))

    return tot_correct/nbatches/mbs, tot_correct_adv/nbatches/mbs, adv_examples, preds, preds_adv, delta_right, delta_wrong

        
def hebbian_syn_grads(model, x, y, neurons, beta, criterion, coeff=1):
    for s in model.synapses:
        s.requires_grad_(True)
    model.zero_grad()
    #phi = model.Phi(x, y, neurons, beta, criterion=criterion)
    hopfield = (neurons[0] * model.synapses[0](x) ).sum() # Energy = y.W.x
    (-coeff*hopfield.sum()).backward() # minimizing optimizer should increase hopfield energy for this configuration, so negative
    for s in model.synapses:
        s.requires_grad_(False)

def compute_lca_syn_grads(model, x, neurons, beta, criterion):
    #for s in model.synapses:
    model.zero_grad()

    layers = [x] + neurons
    pools = [torch.nn.Identity()] + model.pools

    hopfield = 0

    for idx in model.sparse_layer_idxs:
        if idx-1 in model.sparse_layer_idxs:
            target = pools[idx](model.membpotes[idx-1])
        else:
            target = layers[idx]
        if isinstance(model.synapses[idx], torch.nn.Conv2d):
            xhat = F.conv_transpose2d(layers[idx+1], model.synapses[idx].weight, padding=model.synapses[idx].padding, stride=model.synapses[idx].stride).detach()
            model.synapses[idx].requires_grad_(True) # we're using autograd just as a clever way of getting the pre*post neural activity product, attaching the gradient autoamtically to the corresponding synapse
            # when doing this, its essential grad is only used on the following step, and not calculating the reconstruction, or it will include another unrelated term
            # this could also be done with xhat.detach()
            hopfield += (layers[idx+1] * model.synapses[idx](target-xhat) ).sum()
            #model.synapses[idx].requires_grad_(False)
        elif isinstance(model.synapses[idx], torch.nn.Linear):
            xhat = torch.matmul(layers[idx+1], model.synapses[idx].weight).detach()
            model.synapses[idx].requires_grad_(True)
            hopfield += (layers[idx+1] * model.synapses[idx](target.view(x.size(0),-1)-xhat) ).sum()
            #model.synapses[idx].requires_grad_(False)
        # gradient wrt. weights (couretsy of autograd) is the same as the "outer product" (and analogue for conv. nets.) L = s_i w_ij s_j  =>  dL/dw_ij = s_i * s_j
    (-hopfield.sum()).backward() # minimizing optimizer should increase hopfield energy for this configuration, so negative

    #for s in model.synapses:
    with torch.no_grad():
        # average gradient by how much that feature was active across space (2,3) and batches (0)
        for j in model.sparse_layer_idxs:
            if isinstance(model.synapses[j], torch.nn.Conv2d):
                model.synapses[j].weight.grad.div_(neurons[j].norm(0, dim=(0,2,3))[:,None,None,None] + 1e-5) # average gradient by amount the feature was active (activity in latent layer) in the batch
            elif isinstance(model.synapses[j], torch.nn.Linear):
                model.synapses[j].weight.grad.div_(neurons[j].norm(0, dim=0)[:,None] +1e-5)

def compute_lcaep_syn_grads(model, x, y, neurons_1, neurons_2, mp_1, mp_2, betas, criterion):
    mbs = x.size(0)
    #for s in model.synapses:
    model.zero_grad()

    xhat_1 = F.conv_transpose2d(neurons_1[0], model.synapses[0].weight, padding=model.synapses[0].padding, stride=model.synapses[0].stride).detach()
    xhat_2 = F.conv_transpose2d(neurons_2[0], model.synapses[0].weight, padding=model.synapses[0].padding, stride=model.synapses[0].stride).detach()
    # use the reconstruction in the anti-nudged state (1) so this resembles a LCA rule learning to reconstruct
    pot_1 = [x] + mp_1 #[xhat] + 
    pot_2 = [x] + mp_2
    #pot_2 = pot_1 # same features in potential, different activation patterns => should imprint features onto neurons that should be more active in nudge, disimprint the present features into those that don't help
    # makes more sense for two-phase ep (no anti-nudge)
    act_1 = [x] + neurons_1
    act_2 = [x] + neurons_2
    #act_2 = act_1
    #act_2 = act_1 # use only the free state activation. Makes this analogous to STDP where dW = act(s_post) * d(s_pre)/dt
    pools = [torch.nn.Identity()] + model.pools

    dphi = 0

    for idx in range(len(model.synapses)): #model.sparse_layer_idxs:
        if idx in model.sparse_layer_idxs:
            pre_1 = pot_1
            pre_2 = pot_2
            pp = torch.nn.Identity()
        else:
            pre_1 = act_1
            pre_2 = act_2
            if idx < len(model.pools):
                pp = model.pools[idx]
        if idx-1 in model.sparse_layer_idxs:
            p = pools[idx]
        else:
            p = torch.nn.Identity()
        if isinstance(model.synapses[idx], torch.nn.Conv2d):
            model.synapses[idx].requires_grad_(True)
            dphi += ( act_2[idx+1] * pp(model.synapses[idx](p(pre_2[idx]))) ).sum()
            dphi -= ( act_1[idx+1] * pp(model.synapses[idx](p(pre_1[idx]))) ).sum()
            #model.synapses[idx].requires_grad_(False)
        elif isinstance(model.synapses[idx], torch.nn.Linear):
            model.synapses[idx].requires_grad_(True)
            dphi += ( act_2[idx+1] * (model.synapses[idx](p(pre_2[idx]).view(mbs,-1))) ).sum()
            dphi -= ( act_1[idx+1] * (model.synapses[idx](p(pre_1[idx]).view(mbs,-1))) ).sum()
            #model.synapses[idx].requires_grad_(False)
            
    # hybrid of EP, but similar to deep LCA taking the "outer product" of error and activation in the above layer 
    (dphi.sum()/(betas[0]-betas[1])).backward() # minimizing optimizer should strengthen attractor two and weaken attractor one

    #for s in model.synapses:

   # with torch.no_grad():
   #     # average gradient by how much that feature was active across space (2,3) and batches (0)
   #     for j in model.sparse_layer_idxs:
   #         if isinstance(model.synapses[j], torch.nn.Conv2d):
   #             model.synapses[j].weight.grad.div_((neurons_2[j].norm(0, dim=(0,2,3))+neurons_1[j].norm(0, dim=(0,2,3)))[:,None,None,None] + 1e-5) # average gradient by amount the feature was active (activity in latent layer) in the batch
   #         elif isinstance(model.synapses[j], torch.nn.Linear):
   #             model.synapses[j].weight.grad.div_((neurons_2[j].norm(0, dim=0) + neurons_1[j].norm(0, dim=0))[:,None] +1e-5)

def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, alg='EP', 
          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, scheduler=None, cep_debug=False, tensorboard=False, annealcompetition=False, keep_checkpoints=0):
    

    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = betas[:2]
    
    if tensorboard:
        tb_write_freq = iter_per_epochs/3 # every 3 iters update tensorboard
        writer = SummaryWriter('runs/{}/{}/{}'.format(alg, model.__class__.__name__, path))

    if checkpoint is None:
        train_acc = [10.0]
        test_acc = [10.0]
        best = 0.0
        epoch_sofar = 0
        angles = [90.0]
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']
        angles = checkpoint['angles'] if 'angles' in checkpoint.keys() else []

    """
    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)

    pT1 = torch.tensor(1).to(device)
    pT2 = torch.tensor(10).to(device)
    #with record_function("init_neurons", criterion=criterion):
    neurons = model.init_neurons(x.size(0), device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True, record_shapes=True) as prof:
        #with record_function("model_inference"):
        neurons = model(x, y, neurons, pT1, beta=beta_1)
        #neurons_1 = copy(neurons)
        #        #model(x, y, neurons, pT2, beta=beta_2, criterion=criterion)
        #neurons_2 = copy(neurons)
        ##with record_function("synapse_update"):
        #model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

    # prof.export_chrome_trace("trace-{}.json".format(model.__class__))

    print(prof.key_averages(group_by_stack_n=10).table(sort_by="cuda_memory_usage", row_limit=20)) 
    """
    # for reconstruction only, select random portion of input to clamp
    isreconstructmodel = hasattr(model, 'fullclamping')
    #masktransform = torchvision.transforms.Compose([
    #                                  torchvision.transforms.RandomCrop(size=model.in_size, padding=model.in_size//2, padding_mode='constant'),
    #                              ])
    reconstructfreq = 1 # train reconstruct on 1 of x batches
    minreconstructepoch = 0
    maxreconstructepoch = 10
    floatdur = T2
    #isreconstructmodel = issubclass(model.__class__, ReversibleCNN) #False # disable reconstruction trianing

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        recon_err = 0
        model.train()

        if annealcompetition and epoch+epoch_sofar <= 10:
            model.inhibitstrength = (epoch+epoch_sofar)/10
            print('competition annealer: inhibition strength coeff. now = ', model.inhibitstrength)
        if hasattr(model, 'postupdate'):
            model.postupdate()
        if hasattr(model, 'lat_syn'):
            print('lat weight dims', [l.weight.size() for l in model.lat_syn])
            print('lat weight L1 norms', [l.weight.norm(1).item() for l in model.lat_syn])
            if len(model.lat_syn) > 0:
                print(model.lat_syn[-1].weight[0:10,0:10])
        if hasattr(model, 'conv_lat_syn'):
            print('conv lat weight dims', [l.weight.size() for l in model.conv_lat_syn])
            print('conv lat weight L1 norms', [l.weight.norm(1).item() for l in model.conv_lat_syn])
            if len(model.conv_lat_syn) > 0:
                print(model.conv_lat_syn[0].weight[0,0,:,:])
        if hasattr(model, 'conv_comp_layers'):
            print('convolutional competition kernel L1 norms', [l.weight.norm(1) for l in model.conv_comp_layers])
            if len(model.conv_comp_layers) > 0:
                print(model.conv_comp_layers[-1].weight[0,0,:,:])
        if hasattr(model, 'fc_comp_layers'):
            print('fc competition matricies L1', [l.weight.norm(1) for l in model.fc_comp_layers])
            if len(model.fc_comp_layers) > 0:
                print(model.fc_comp_layers[-1].weight[0:10,0:10])
        print('synapse L2 norms', [l.weight.norm(2).item() for l in model.synapses])
        print('bias L1 norms', [l.bias.norm(1).item() for l in model.synapses])
        print('final forward fc bias', model.synapses[-1].bias)

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            mbs = x.size(0)

            reconstruct = isreconstructmodel #(model.__class__.__name__ == 'LCACNN') #isreconstructmodel and int(idx%reconstructfreq) == 0 and epoch >= minreconstructepoch and epoch < maxreconstructepoch
            classify = not isreconstructmodel and not (alg == "LCA")

            neurons = model.init_neurons(mbs, device)

            #if alg=='CEP' and cep_debug:
            #    x = x.double() 
            if reconstruct:
                # results in a zero and one mask used to select which indexes of the input to fully clamp.
                # The zeros (the padding included in the randomcrop) are where it will reconstruct the output
                #reconstructmask = torch.stack([masktransform(torch.ones_like(x[0])) for i in range(mbs)])
                #model.fullclamping[0] = reconstructmask.bool()
                classifyalso = False#torch.rand((1,)).item() < 0.1 # this percent of the time, simulataneoulsy nudge the classification
                #model.mode(trainclassifier=classifyalso, trainreconstruct=True)#, noisebeta=True)#(torch.randn((1,)).item() < 0.5))
                
                origx = x.clone()
                #origx = x
                if torch.randn((1,)).item() < 0.5:
                    x = AddGaussianNoise(0.0, 0.2)(x)

                #neurons = model(x, y, neurons, T1, beta_2)
                #print(neurons)
                #hebbian_syn_grads(model, x, y, neurons, beta_2, criterion, 1)
                #optimizer.step()
                #neurons = model.init_neurons(mbs, device)


                if model.__class__.__name__ == 'sparseCNN' :#torch.rand((1,)).item() < 0.1:
                    model.mode(trainclassifier=False, trainreconstruct=True)
                    #if alg == 'EP': # rather than nudging output, impose L1 penalty to nudge for sparsity
                    #    with torch.no_grad():
                    #        for layer in model.conv_comp_layers: # L1 penalty is equivelant to subtracting lambda from biases
                    #            layer.bias -= model.lambdas[1]
                    #    
                    #    neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                    #    neurons_1 = copy(neurons)

                    #    neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                    #    neurons_2 = copy(neurons)

                    #    # reset to normal weights
                    #    with torch.no_grad():
                    #        for layer in model.conv_comp_layers:
                    #            layer.bias += model.lambdas[1]

                    #    # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                    #    if thirdphase:
                    #        # anti-nudge: promote non-sparsity with more positive biases
                    #        with torch.no_grad():
                    #            for layer in model.conv_comp_layers:
                    #                layer.bias += model.lambdas[1]
                    #        #come back to the first equilibrium
                    #        neurons = copy(neurons_1)
                    #        neurons = model(x, y, neurons, T2, beta=-beta_2, criterion=criterion)
                    #        neurons_3 = copy(neurons)
                    #        # reset bias values
                    #        with torch.no_grad():
                    #            for layer in model.conv_comp_layers:
                    #                layer.bias -= model.lambdas[1]
                    #        #if not(isinstance(model, VF_CNN)):
                    #        model.compute_syn_grads_sparsity(x, y, neurons_2, neurons_3, beta_1, (model.lambdas[1], -model.lambdas[1]), criterion)
                    #    else:
                    #        model.compute_syn_grads_sparsity(x, y, neurons_1, neurons_2, beta_1, model.lambdas, criterion)

                    #    model.sparse_optim.step()
                    #    if hasattr(model, 'postupdate'):
                    #        model.postupdate()
                    #    
                    #    print('\tsparse nudge:')
                    #    print('\t\tneurons_1 L1', [(neurons_1[layeridx]).norm(1).item() for layeridx in range(len(neurons_1))])
                    #    print('\t\tneurons_2 L1', [(neurons_2[layeridx]).norm(1).item() for layeridx in range(len(neurons_1))])
                    #    if thirdphase:
                    #        print('\t\tneurons_3 L1', [(neurons_3[layeridx]).norm(1).item() for layeridx in range(len(neurons_1))])
                    #        print('\t\tneurons_2-neurons_3 L2', [(neurons_2[layeridx]-neurons_3[layeridx]).norm(2).item() for layeridx in range(len(neurons_1))])
                    #    print('\t\tneurons_1-neurons_2 L2', [(neurons_2[layeridx]-neurons_1[layeridx]).norm(2).item() for layeridx in range(len(neurons_1))])
                    #    print('\tsparse lat fc connections :', model.fc_comp_layers[-1].weight.cpu().detach())
                    #    print('\tsparse lat fc connections bias:', model.fc_comp_layers[-1].bias.cpu().detach())
                    #    print('\tsparse lat conv connections bias:', model.conv_comp_layers[-1].bias.cpu().detach())

                    #    continue
                    
            if isreconstructmodel:
                #model.mode(trainclassifier=True, trainreconstruct=False)
                model.fullclamping[0].fill_(True)
                #model.fullclamping[0].fill_(False)

                        
            # run sparse training step
            #if isinstance(model, sparseCNN):
            #    if torch.rand((1,)).item() < 0.2: # train sparsity for 20% of batches
            #        #model.train_sparsity_EP(x, y, T, thirdphase=thirdphase)
            #        if alg == 'EP':
            #            neurons = copy(neurons_1)
            #            model.lambda_val = model.lambdas[1]
            #            neurons = model(x, y, T2, beta=beta_1, criterion=criterion)
            #        model.lambda_val = model.lambdas[0]
        

            if alg=='EP' or alg=='CEP':
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
            elif alg=="LCAEP":
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
                mp_1 = copy(model.membpotes)
            if alg == "LCA":
                neurons = model(x, y, neurons, T1, beta=0.0)
                compute_lca_syn_grads(model, x, neurons, 0.0, criterion)

                        
                optimizer.step()
                if hasattr(model, 'postupdate'):
                    model.postupdate()                

                for j, g in enumerate(optimizer.param_groups):
                    if g['lr'] == 0.0: # decoupling other layers we don't want to interfere in the LCA training phase
                        model.synapses[j].weight.zero_() # fc layer required based on how code is written, but don't want it to interfere
                        model.synapses[j].bias.zero_() # fc layer required based on how code is written, but don't want it to interfere
            if alg =="LCA" or alg=="LCAEP":
                xhat = F.conv_transpose2d(neurons[0], model.synapses[0].weight, padding=model.synapses[0].padding, stride=model.synapses[0].stride)

                recon_err = ( (x - xhat).norm(2, dim=(1,2,3)).mean() ).data.item()

                if tensorboard:
                    batchiter = (epoch_sofar+epoch)*iter_per_epochs+idx
                    im = lambda t: ((t+1)/2*255).to(torch.uint8)

                    img_grid = torchvision.utils.make_grid(im(x[:16].data.cpu()))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('original input', img_grid, batchiter)

                    img_grid = torchvision.utils.make_grid(im(xhat[:16].data.cpu()))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('completed input layer, phase 1', img_grid, batchiter)
                
                    writer.add_scalars('reconstruct', {'recon_err': recon_err,}, batchiter)

                    writer.close()
            elif alg=='BPTT':
                neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion)           
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, T2, beta=0.0, check_thm=True, criterion=criterion) # T2 time step

            # Predictions for running accuracy
            with torch.no_grad():
                if not model.softmax:
                    pred = torch.argmax(neurons[-1], dim=1).squeeze()
                else:
                    #WATCH OUT: prediction is different when softmax == True
                    pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

                run_correct += (y == pred).sum().item()
                run_total += mbs # x.size(0)
                    
            if reconstruct:
                # let the input float phase
                #print('before float', [n.norm(2).item() for n in neurons_1])
                with torch.no_grad():
                    #neurons_1[0].zero_()
                    neurons_1[0] = F.conv_transpose2d(neurons_1[1], model.synapses[0].weight, padding=model.synapses[0].padding)
                #model.fullclamping[0].fill_(False)
                #neurons_1 = model(torch.zeros_like(x), y, neurons_1, floatdur, beta=0.0)
                #print('after float ', [n.norm(2).item() for n in neurons_1])
            
            
            if alg=='EP':
                # Second phase
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas[1] = rnd_sgn*beta_2
                    beta_2 = betas[1]
            
                if reconstruct:
                    #x = origx # remove noise, to nudge for denoising
                    model.fullclamping[0].fill_(True)
                    neurons = model(origx, y, neurons, T2, beta = beta_2, criterion=criterion)
                else:
                    neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)

                # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if thirdphase:
                    if len(betas) > 2:
                        beta_3 = betas[2]
                    else:
                        beta_3 = -beta_2
                    #come back to the first equilibrium
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta = beta_3, criterion=criterion)
                    neurons_3 = copy(neurons)
                    if not(isinstance(model, VF_CNN)):
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, beta_3), criterion)
                    else:
                        if model.same_update:
                            model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, beta_3), criterion)
                        else:    
                            model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, beta_3), criterion, neurons_3=neurons_3)
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas[:2], criterion)

                #print('grads mean', [l.weight.grad.mean().item() for l in model.synapses], 'grads max', [l.weight.grad.max().item() for l in model.synapses], 'grads min', [l.weight.grad.min().item() for l in model.synapses])
                optimizer.step()      
                if hasattr(model, 'postupdate'):
                    model.postupdate()

            if alg=="LCAEP":
                neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                mp_2 = copy(model.membpotes)
                
                if thirdphase:
                    if len(betas) > 2:
                        beta_3 = betas[2]
                    else:
                        beta_3 = -beta_2
                    neurons = copy(neurons_1)
                    model.membpotes = copy(mp_1)
                    neurons = model(x, y, neurons, T2, beta = beta_3, criterion=criterion)
                    neurons_3 = copy(neurons)
                    mp_3 = copy(model.membpotes)
                    #model.compute_syn_grads(x, y, mp_2, mp_3, (beta_2, beta_3), criterion)
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, beta_3), criterion)
                    #compute_lcaep_syn_grads(model, x, y, neurons_2, neurons_3, mp_2, mp_3, (beta_2, beta_3), criterion)
                else:
                    #model.compute_syn_grads(x, y, mp_1, mp_2, (beta_1, beta_2), criterion)
                    model.compute_syn_grads(x, y, neurons_1, nuerons_2, (beta_1, beta_2), criterion)
                    #compute_lcaep_syn_grads(model, x, y, neurons_1, neurons_2, mp_1, mp_2, betas[:2], criterion)
                optimizer.step()      
                #if hasattr(model, 'postupdate'):
                #    model.postupdate()
                
            elif alg=='CEP':
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas

                # second phase
                if cep_debug:
                    prev_p = {}
                    for (n, p) in model.named_parameters():
                        prev_p[n] = p.clone().detach()
                    for i in range(len(model.synapses)):
                        prev_p['lrs'+str(i)] = optimizer.param_groups[i]['lr']
                        prev_p['wds'+str(i)] = optimizer.param_groups[i]['weight_decay']
                        optimizer.param_groups[i]['lr'] *= 6e-5
                        #optimizer.param_groups[i]['weight_decay'] = 0.0
                                        
                for k in range(T2):
                    neurons = model(x, y, neurons, 1, beta = beta_2, criterion=criterion)   # one step
                    neurons_2  = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)   # compute cep update between 2 consecutive steps 
                    for (n, p) in model.named_parameters():
                        p.grad.data.div_( (1 - optimizer.param_groups[int(n[9])]['lr']*optimizer.param_groups[int(n[9])]['weight_decay'])**(T2-1-k)  ) 
                    optimizer.step()                                                        # update weights 
                    neurons_1 = copy(neurons)  
               
                if cep_debug:
                    debug(model, prev_p, optimizer)
 
                if thirdphase:    
                    neurons = model(x, y, neurons, T2, beta = 0.0, criterion=criterion)     # come back to s*
                    neurons_2 = copy(neurons)
                    for k in range(T2):
                        neurons = model(x, y, neurons, 1, beta = -beta_2, criterion=criterion)
                        neurons_3 = copy(neurons)
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
                        optimizer.step()
                        neurons_2 = copy(neurons)

            elif alg=='BPTT':
         
                # final loss
                if criterion.__class__.__name__.find('MSE')!=-1:
                    loss = 0.5*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax:
                        loss = criterion(neurons[-1].float(), y).mean().squeeze()
                    else:
                        loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).mean().squeeze()
                # setting gradients field to zero before backward
                model.zero_grad()

                # Backpropagation through time
                loss.backward()
                optimizer.step()

            if reconstruct:
                recon_err = ( (origx - neurons_1[0]).norm(2)).data.item()

                if tensorboard:
                    batchiter = (epoch_sofar+epoch)*iter_per_epochs+idx
                    im = lambda t: ((t+1)/2*255).to(torch.uint8)

                    img_grid = torchvision.utils.make_grid(im(x.data.cpu()[:16]))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('original input', img_grid, batchiter)

                    model(x, y, neurons, beta=beta_1, T=0) # make it set up targetneurons
                    img_grid = torchvision.utils.make_grid(im((model.targetneurons[0])[:16].data.cpu()))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('input force, phase 1', img_grid, batchiter)

                    img_grid = torchvision.utils.make_grid(im(neurons_1[0][:16].data.cpu()))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('completed input layer, phase 1', img_grid, batchiter)
                
                    model(origx, y, neurons, beta=beta_2, T=0) # make it set up targetneurons
                    img_grid = torchvision.utils.make_grid(im((model.targetneurons[0])[:16].data.cpu()))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('input force, phase 2', img_grid, batchiter)

                    img_grid = torchvision.utils.make_grid(im(neurons_2[0][:16].data.cpu()))
                    # matplotlib_imshow(img_grid, one_channel=True)
                    writer.add_image('completed input layer, phase 2 (hebbian)', img_grid, batchiter)

                    if thirdphase:
                        img_grid = torchvision.utils.make_grid(im(neurons_3[0][:16].data.cpu()))
                        # matplotlib_imshow(img_grid, one_channel=True)
                        writer.add_image('completed input layer, phase 3 (antihebbian)', img_grid, batchiter)


                    writer.add_scalars('reconstruct', {'recon_err': recon_err,}, batchiter)

                    writer.close()

            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch_sofar+epoch+(idx/iter_per_epochs), 2),
                      '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                      timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                if alg == 'EP' or alg == 'CEP' or alg=="LCAEP":
                    print('\tL2 neurons_1-neurons_2 : ', [(neurons_1[idx]-neurons_2[idx]).norm(2).item() for idx in range(len(neurons_1))])
                    if thirdphase:
                        print('\tL2 neurons_1-neurons_3 : ', [(neurons_1[idx]-neurons_3[idx]).norm(2).item() for idx in range(len(neurons_1))])
                        print('\tL2 neurons_2-neurons_3 : ', [(neurons_2[idx]-neurons_3[idx]).norm(2).item() for idx in range(len(neurons_1))])
                if isreconstructmodel or alg=="LCA" or alg=="LCAEP":
                    print('\tReconstruction error (L2) :\t', recon_err)
                if isinstance(model, VF_CNN): 
                    angle = model.angle()
                    print('angles ',angle)
                if check_thm and alg!='BPTT':
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion, alg=alg)
                    RMSE(BPTT, EP)
                if reconstruct:
                    # test denoising
                    noisify = AddGaussianNoise(0.0, 0.1)
                    noisyx = noisify(origx)
                    neurons = model.init_neurons(origx.size(0), origx.device)
                    model.fullclamping[0].fill_(True)
                    neurons = model(noisyx, y, neurons, T1, beta_1)
                    with torch.no_grad():
                        #tneurons_1[0].zero_()
                        neurons[0] = F.conv_transpose2d(neurons[1], model.synapses[0].weight, padding=model.synapses[0].padding)
                    model.fullclamping[0].fill_(False)
                    neurons = model(noisyx, y, neurons, T2, beta_2)

                    denoise_err = (neurons[0] - origx).norm(2, dim=(1,2,3)).mean().item()

                    if tensorboard:
                        batchiter = (epoch_sofar+epoch)*iter_per_epochs+idx
                        writer.add_scalars('denoising', {'recon vs clean err': denoise_err,}, batchiter)
                if alg=="LCA" or alg=="LCAEP":
                    noisify = AddGaussianNoise(0.0, 0.1)
                    noisyx = noisify(x)
                    neurons = model.init_neurons(mbs, device)
                    neurons = model(noisyx, y, neurons, T1, beta_1)
                    xhat = F.conv_transpose2d(neurons[0], model.synapses[0].weight, padding=model.synapses[0].padding)

                    denoise_err = (xhat - x).norm(2, dim=(1,2,3)).mean().item()

                    if tensorboard:
                        batchiter = (epoch_sofar+epoch)*iter_per_epochs+idx
                        writer.add_scalars('denoising', {'recon vs clean err': denoise_err,}, batchiter)
                    

                if save:
                    # plot how much the neurons are changing to know when it equilibrates
                    neurons = model.init_neurons(mbs, device)
                    l2s = [[] for i in range(len(neurons))]
                    phis = []
                    for t in range(T1):
                        lastneurons = copy(neurons)
                        neurons = model(x, y, neurons, 1, beta=beta_1, criterion=criterion)
                        [l2s[layeridx].append((neurons[layeridx]-lastneurons[layeridx]).norm(2).item()) for layeridx in range(len(l2s))]
                        if not isreconstructmodel:
                            phis.append(model.Phi(x, y, neurons, beta=beta_1, criterion=criterion).sum().item())
                    if reconstruct:
                        with torch.no_grad():
                            #neurons_1[0].zero_()
                            neurons[0] = F.conv_transpose2d(neurons[1], model.synapses[0].weight, padding=model.synapses[0].padding)
                        #model.fullclamping[0].fill_(False)
                        for t in range(floatdur):
                            lastneurons = copy(neurons)
                            neurons = model(x, y, neurons, 1, beta=0.0, criterion=criterion)
                            [l2s[layeridx].append((neurons[layeridx]-lastneurons[layeridx]).norm(2).item()) for layeridx in range(len(l2s))]
                            if not isreconstructmodel:
                                phis.append(model.Phi(x, y, neurons, beta=beta_1, criterion=criterion).sum().item())
                    # also plot histogram of neuron values
                    plot_neural_activity(neurons, path, suff=epoch)
                    for t in range(T2):
                        lastneurons = copy(neurons)
                        neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)
                        [l2s[layeridx].append((neurons[layeridx]-lastneurons[layeridx]).norm(2).item()) for layeridx in range(len(l2s))]
                        if not isreconstructmodel:
                            phis.append(model.Phi(x, y, neurons, beta=beta_1, criterion=criterion).sum().item())
                    plot_neural_activity(neurons, path, suff=str(epoch_sofar+epoch)+'_nudged')
                    N = len(neurons)
                    fig = plt.figure(figsize=(3*N,6))
                    for layeridx in range(N):
                        fig.add_subplot(2, N//2+1, layeridx+1)
                        if reconstruct:
                            plt.plot(range(T1+floatdur+T2), l2s[layeridx])
                        else:
                            plt.plot(range(T1+T2), l2s[layeridx])
                        plt.title('L2 change in neurons of layer '+str(layeridx+1))
                        plt.xlabel('time step')
                        plt.yscale('log')
                        plt.axvline(x=T1, linestyle=':')
                        plt.tight_layout()
                    fig.savefig(path + '/neural_equilibrating_{}.png'.format(epoch_sofar+epoch), bbox_inches='tight')
                    plt.close()

                    if not isreconstructmodel:
                        fig = plt.figure()
                        plt.plot(range(T1+T2), phis)
                        plt.title('Energy Function (Phi) over Model Dynamics Evolution')
                        plt.xlabel('time step')
                        plt.ylabel('energy')
                        plt.axvline(x=T1, linestyle=':')
                        plt.tight_layout()
                        fig.savefig(path + '/phi_evolution_{}.png'.format(epoch_sofar+epoch), bbox_inches='tight')
                        plt.close()

            
            if tensorboard:
                if ((idx%(iter_per_epochs//tb_write_freq)==0) or (idx==iter_per_epochs-1)):
                    batchiter = (epoch_sofar+epoch)*iter_per_epochs+idx

                    run_acc = run_correct/run_total
                    writer.add_scalars('accuracy', {'train_acc': run_acc,}, batchiter)

                    writer.close()

        if scheduler is not None: # learning rate decay step
            if epoch+epoch_sofar < scheduler.T_max:
                scheduler.step()
        if hasattr(model, 'sparse_lr_scheduler'):
            if epoch+epoch_sofar < model.sparse_lr_scheduler.T_max:
                model.sparse_lr_scheduler.step()

        test_correct = 0
        if classify:
            test_correct = evaluate(model, test_loader, T1, device)
        test_acc_t = test_correct/(len(test_loader.dataset))
        run_acc = run_correct/run_total
        if tensorboard:
            batchiter = (epoch_sofar+epoch)*iter_per_epochs+idx
            writer.add_scalars('accuracy', {'test_acc': test_acc_t,}, batchiter)
            writer.close()
        if save:
            test_acc.append(100*test_acc_t)
            train_acc.append(100*run_acc)
            if isinstance(model, VF_CNN):
                angle = model.angle()
                angles.append(angle)
            save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                        'train_acc': train_acc, 'test_acc': test_acc, 
                        'best': best, 'epoch': epoch_sofar+epoch+1}
            if hasattr(model, 'sparse_optim'):
                save_dic['sparse_optim'] = model.sparse_optim.state_dict()
            save_dic['angles'] = angles
            save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
            torch.save(save_dic,  path + '/checkpoint.tar')
            torch.save(model, path + '/model.pt')
            if keep_checkpoints > 0 and epoch% keep_checkpoints == 0:
                torch.save(model, path + '/model_{}.pt'.format(epoch+epoch_sofar))
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1}
                if hasattr(model, 'sparse_optim'):
                    save_dic['sparse_optim'] = model.sparse_optim.state_dict()
                save_dic['angles'] = angles
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  path + '/best_checkpoint.tar')
                torch.save(model, path + '/best_model.pt')
            plot_acc(train_acc, test_acc, path)        

    
    if save:
        save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                    'train_acc': train_acc, 'test_acc': test_acc, 
                    'best': best, 'epoch': epoch_sofar+epochs}
        if hasattr(model, 'sparse_optim'):
            save_dic['sparse_optim'] = model.sparse_optim.state_dict()
        save_dic['angles'] = angles
        save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
        torch.save(save_dic,  path + '/final_checkpoint.tar')
        torch.save(model, path + '/final_model.pt')
 

            
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


            










