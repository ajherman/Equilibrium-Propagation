import torch
from model_utils import * 
from lateral_models import *

import typing

class Reversible_CNN(P_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=my_sigmoid, softmax=False, inputbeta=0.5, dt=0.5):
        P_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax)

        self.fullclamping = []
        for idx in range(len(self.synapses)+1):
            self.fullclamping.append(torch.tensor([]))

        self.mode(trainclassifier=True, trainreconstruct=False)

        self.reconcriterion = torch.nn.MSELoss(reduction = 'none')

        self.inputbeta = inputbeta

        self.dt = dt

    def mode(self, trainclassifier=True, trainreconstruct=False):
        self.trainclassifier = trainclassifier
        self.trainreconstruct = trainreconstruct

    def postupdate(self):
        with torch.no_grad():
            #self.synapses[0].bias.fill_(-0.35)
            self.synapses[0].weight.data /= self.synapses[0].weight.norm(2, dim=(0,2,3))[None,:,None,None]

    def Phi(self, targetneurons, neurons, betas, fullclamping, criterion):

        mbs = neurons[0].size(0)
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        #layers = [x] + neurons        
        #layers = neurons
        phi = 0.0

        for idx in range(conv_len):    
            phi += torch.sum( self.pools[idx](self.synapses[idx](neurons[idx])) * neurons[idx+1], dim=(1,2,3)).squeeze()     
        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len, tot_len):
                phi += torch.sum( self.synapses[idx](neurons[idx].view(mbs,-1)) * neurons[idx+1], dim=1).squeeze()
        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len, tot_len-1):
                phi += torch.sum( self.synapses[idx](neurons[idx].view(mbs,-1)) * neurons[idx+1], dim=1).squeeze()
             
        if targetneurons[0].size(0) > 0:
            # MSE with the input, like nudge
            phi -= 0.5*(betas[0]*self.reconcriterion(neurons[0], targetneurons[0])).view(mbs,-1).sum(dim=1).squeeze() 
            # dot-product energy, like Identity input layer
            #phi += torch.sum(neurons[0]*targetneurons[0])
        for idx in range(1,len(neurons)):
            if targetneurons[idx].size(0) > 0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    L = 0.5*(betas[idx]*criterion(neurons[idx], targetneurons[idx])).view(mbs,-1).sum(dim=1).squeeze() 
                else:
                    L = (betas[idx]*criterion(neurons[idx], targetneurons[idx])).view(mbs,-1).sum(dim=1).squeeze() 
                phi -= L    

        # for idx in range(len(neurons)):
        #     if fullclamping[idx].size(0) > 0:
        #         phi -= 0.2*neurons[idx][False == fullclamping[idx]].norm(1) # L1 penalty for lots of activation in reconstruction
        return phi
    
    def xytotargetneurons(self, x, y):
        # targetneurons = torch.zeros_like(neurons)
        for idx in range(len(self.targetneurons)):
            self.targetneurons[idx].zero_()
        self.targetneurons[0].copy_(x)
        self.targetneurons[-1].copy_(F.one_hot(y.to(torch.int64), num_classes=self.nc))

        return self.targetneurons

    def fieldbeta(self, beta):
        for idx in range(len(self.betas)):
            self.betas[idx].zero_()
        # nudge on input for reconstruction of unclamped portion
        if self.trainreconstruct:
            self.betas[0].fill_(self.inputbeta + beta)
        else:
            self.betas[0].fill_(self.inputbeta)
        # nudge for classification
        if self.trainclassifier:
            self.betas[-1].fill_(beta)

        return self.betas

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
 
        # not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     

        self.targetneurons = self.xytotargetneurons(x, y)
        self.betas = self.fieldbeta(beta)

        return self.energymodel(self.targetneurons, neurons, T, self.betas, self.fullclamping, criterion, check_thm=check_thm)

    def energymodel(self, targetneurons, neurons, T, betas, fullclamping, criterion, check_thm=False):
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = neurons[0].size(0)
        device = neurons[0].device

        # apply full clamping
#        with torch.no_grad():
        for idx in range(len(neurons)):
#                if fullclamping[idx].size(0) >  0:
#                    neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]
            if neurons[idx].is_leaf:
                neurons[idx].requires_grad = True

        # simulate dynamics for T timesteps
        #print('weight', self.synapses[0].weight.norm(1, dim=(0,2,3)))
        for t in range(T):
            #with torch.no_grad():
            #    neurons[1].zero_()
            phi = self.Phi(targetneurons, neurons, betas, fullclamping, criterion)
            grads = torch.autograd.grad(phi, neurons, grad_outputs=self.initgradstensor, create_graph=check_thm)
            
            #neurons[0] = -2 + 4*self.activation( ((1-self.dt)*neurons[0] + self.dt*grads[0]) / 4 + 0.5 )
            #print('x', targetneurons[0].min().item(), targetneurons[0].mean().item(), targetneurons[0].max().item(), 'LGN', neurons[0].min().item(), neurons[0].mean().item(), neurons[0].max().item(), 'grads', grads[0].min().item(), grads[0].mean().item(), grads[0].max().item(), 'grads idx+1', grads[1].mean().item(), 'layer idx+1', neurons[1].mean().item())

    
            neurons[0] = ((1-self.dt)*neurons[0] + self.dt*grads[0])
            #neurons[0].requires_grad = True

            for idx in range(1,len(neurons)-1):
                neurons[idx] = self.activation( (1-self.dt)*neurons[idx] + self.dt*grads[idx] )
                #neurons[idx].requires_grad = True

            # apply full clamping
#            with torch.no_grad():
#                for idx in range(len(neurons)):
#                    if fullclamping[idx].size(0) >  0:
#                        neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]
#
            ## average the last n neurons in case its a limit cycle

        return neurons
       
    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        append(torch.zeros((mbs, self.channels[0], size, size), requires_grad=True, device=device))
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
            
        self.targetneurons = []
        self.betas = []
        self.fullclamping = []
        for idx in range(len(neurons)):
            neurons[idx].requires_grad_(False)
            self.targetneurons.append(torch.zeros_like(neurons[idx]))#tensor([], device=device))
            self.betas.append(torch.tensor([0.0], device=device))
            self.fullclamping.append(torch.zeros_like(neurons[idx]).bool())
        # for most layers targetneurons isn't being used, so use an empty tensor rather than zeros.
        # but for input and output layer, it should always have the right size
        self.targetneurons[0].data = torch.zeros_like(neurons[0])
        self.targetneurons[-1].data = torch.zeros_like(neurons[-1])
        self.betas[0].data = torch.zeros((1,), device=device)
        self.betas[-1].data = torch.zeros((1,), device=device)

        self.fullclamping[0] = torch.ones_like(neurons[0]).bool()

        self.initgradstensor = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)

        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, beta1beta2, criterion, check_thm=False):
        beta_1, beta_2 = beta1beta2

        self.targetneurons = self.xytotargetneurons(x, y)

        betas1 = copy(self.fieldbeta(beta_1))
        betas2 = copy(self.fieldbeta(beta_2))

        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(self.targetneurons, neurons_1, betas1, self.fullclamping, criterion)
        else:
            phi_1 = self.Phi(self.targetneurons, neurons_1, betas2, self.fullclamping, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(self.targetneurons, neurons_2, betas2, self.fullclamping, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem



class RevLatCNN(Reversible_CNN, lat_CNN, torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=my_sigmoid, softmax=False, inputbeta=0.5, dt=0.5):
        softmax = False
        # super(RevLatCNN, self)
        # note - order of super inits called is important for some reason.
        Reversible_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax, inputbeta=inputbeta, dt=dt)
        self.call_super_init = False # prevents pytorch from clearing all the paramters added by the first init
        lat_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=activation, softmax=softmax)
        print('sll', self.lat_layer_idxs, self.lat_syn)
        for i in range(len(self.lat_layer_idxs)):
            self.lat_layer_idxs[i] += 1 # input layer is now in system, shift indexes by one
        """ # will be done by super with multiple inheritance
        self.lat_layer_idxs = lat_layer_idxs
        for i in range(len(self.lat_layer_idxs)):
            while self.lat_layer_idxs[i] <= 0:
                self.lat_layer_idxs[i] += len(self.synapses)
        
        self.lat_syn = torch.nn.ModuleList()
        size = in_size
        
        # layers have already been added by super(), just compute the size
        for idx in range(len(channels)-1): 
#             self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
#                                                  stride=strides[idx], padding=paddings[idx], bias=True))
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool
            if idx+1 in self.lat_layer_idxs:
                fcsize = size * size * channels[idx+1]
                self.lat_syn.append(torch.nn.Linear(fcsize, fcsize, bias=True))

        size = size * size * channels[-1]        
        # fully connect it back down to output dimension
        fc_layers = [size] + fc
        for idx in range(len(fc)):
            # self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
            if len(channels) + idx in self.lat_layer_idxs:
                self.lat_syn.append(torch.nn.Linear(fc_layers[idx+1], fc_layers[idx+1], bias=True))

        """

    def Phi(self, targetneurons, neurons, betas, fullclamping, criterion):
        phi = Reversible_CNN.Phi(self, targetneurons, neurons, betas, fullclamping, criterion)
        mbs = neurons[0].size(0)

        for j, idx in enumerate(self.lat_layer_idxs):
            phi += torch.sum( self.lat_syn[j](neurons[idx].view(mbs,-1)) * neurons[idx].view(mbs,-1), dim=1).squeeze()

        return phi


class RevConvLatCNN(Reversible_CNN, lat_conv_CNN, torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_kernels, lat_layer_idxs, lat_constraints, activation=my_sigmoid, softmax=False, inputbeta=0.5, dt=0.5):
        softmax = False
        # super(RevLatCNN, self)
        # note - order of super inits called is important for some reason.
        Reversible_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax, inputbeta=inputbeta, dt=dt)
        self.call_super_init = False # prevents pytorch from clearing all the paramters added by the first init
        lat_conv_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_kernels, lat_layer_idxs, lat_constraints, activation=activation, softmax=softmax)
        print('sll', self.lat_layer_idxs, self.lat_syn)
        for i in range(len(self.lat_layer_idxs)):
            self.lat_layer_idxs[i] += 1 # input layer is now in system, shift indexes by one
        for i in range(len(self.conv_lat_layer_idxs)):
            self.conv_lat_layer_idxs[i] += 1 # input layer is now in system, shift indexes by one

    def Phi(self, targetneurons, neurons, betas, fullclamping, criterion):
        phi = Reversible_CNN.Phi(self, targetneurons, neurons, betas, fullclamping, criterion)
        mbs = neurons[0].size(0)
        
        for i, idx in enumerate(self.conv_lat_layer_idxs):
            phi += torch.sum( self.conv_lat_syn[i](neurons[idx]) * neurons[idx] , dim=(1,2,3) ).squeeze()

        for j, idx in enumerate(self.lat_layer_idxs):
            phi += torch.sum( self.lat_syn[j](neurons[idx].view(mbs,-1)) * neurons[idx].view(mbs,-1), dim=1).squeeze()

        return phi




class RevLCACNN(Reversible_CNN, latCompCNN, torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, inhibitstrength, competitiontype, lat_layer_idxs, lat_constraints, sparse_layer_idxs=[-1], comp_syn_constraints=['zerodiag+transposesymmetric'], activation=hard_sigmoid, softmax=False, layerlambdas=[1.0e-2], inputbeta=0.5, dt=1.0):
        softmax = False
        # super(RevLatCNN, self)
        # note - order of super inits called is important for some reason.
        Reversible_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax, inputbeta=inputbeta, dt=dt)
        self.call_super_init = False # prevents pytorch from clearing all the paramters added by the first init
        latCompCNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, inhibitstrength, competitiontype, lat_layer_idxs, lat_constraints, sparse_layer_idxs, comp_syn_constraints, activation=activation, softmax=softmax, layerlambdas=layerlambdas, dt=dt)
        print('sll', self.lat_layer_idxs, self.lat_syn)
        for i in range(len(self.lat_layer_idxs)):
            self.lat_layer_idxs[i] += 1 # input layer is now in system, shift indexes by one
#        for i in range(len(self.conv_lat_layer_idxs)):
#            self.conv_lat_layer_idxs[i] += 1 # input layer is now in system, shift indexes by one
#        for i in range(len(self.sparse_layer_idxs)):
#            self.sparse_layer_idxs[i] += 1 # input layer is now in system, shift indexes by one

    def Phi(self, targetneurons, neurons, betas, fullclamping, criterion):
        phi = Reversible_CNN.Phi(self, targetneurons, neurons, betas, fullclamping, criterion)
        mbs = neurons[0].size(0)
        
#        for i, idx in enumerate(self.conv_lat_layer_idxs):
#            phi += torch.sum( self.conv_lat_syn[i](neurons[idx]) * neurons[idx] , dim=(1,2,3) ).squeeze()

        for j, idx in enumerate(self.lat_layer_idxs):
            phi += torch.sum( self.lat_syn[j](neurons[idx].view(mbs,-1)) * neurons[idx].view(mbs,-1), dim=1).squeeze()

        for j, layer in enumerate(self.conv_comp_layers):
            idx = self.sparse_layer_idxs[j] + 1
            phi += torch.sum(layer(neurons[idx]) * neurons[idx], dim=(1,2,3))

        for j, layer in enumerate(self.fc_comp_layers):
            idx = self.sparse_layer_idxs[j+len(self.conv_comp_layers)] + 1
            phi += torch.sum(layer(neurons[idx]) * neurons[idx], dim=1)

        return phi
# absolute disgusting kludge. why do I have to do this in pytorch just to use JIT with cuda
# add ModuleInterface
@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass

# input-reconstructing, lateral-connectivity, analytical equation of motion, hopfield energy model
@torch.jit.interface
class MyModuleInterface(torch.nn.Module):
    def xytotargetneurons(self, x, y) -> torch.Tensor:
        pass

class HopfieldCNN(RevLatCNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=my_sigmoid, softmax=False, criterion=torch.nn.CrossEntropyLoss(reduction='none')):
        RevLatCNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=activation, softmax=False)
        self.in_size = torch.jit.annotate(int, self.in_size)

        self.criterion = criterion

        # define transposes of each layer to send spikes backwards
        #self.syn_transpose = torch.nn.ModuleList()
        #for idx in range(len(self.synapses)):
        #    layer = self.synapses[idx]
        #    if isinstance(layer, torch.nn.Conv2d):
        #        print()
        #    elif isinstance(layer, torch.nn.Linear):
        #        transpose = torch.nn.Linear(layer.out_features, layer.in_features, bias=False)
        #        del transpose.weight # should be tied to forward weights, in postupdate()
        #    self.syn_transpose.append(transpose)
        self.postupdate()

        # reverse pooling operations
        self.unpools = torch.jit.annotate(torch.nn.ModuleList, torch.nn.ModuleList())
        self.unpooldata = torch.jit.annotate(typing.List[torch.Tensor], [])
        for idx in range(len(self.pools)):
            pool = self.pools[idx]
            self.unpooldata.append(torch.tensor([]))
            if isinstance(pool, torch.nn.MaxPool2d):
                self.unpools.append(torch.nn.MaxUnpool2d(pool.kernel_size, stride=pool.stride)) 
            elif isinstance(pool, torch.nn.AvgPool2d):
                if isinstance(pool.kernel_size, tuple):
                    kern0 = pool.kernel_size[0]
                    kern1 = pool.kernel_size[1]
                else:
                    kern0 = kern1 = pool.kernel_size
                avgunpool = torch.nn.Conv2d(1, 1, (kern1,kern0), stride=1, padding=(kern1-1,kern0-1), dilation=pool.stride, bias=False)
                avgunpool.weight.requires_grad = False
                avgunpool.weight.data = avgunpool.weight.fill_(1/pool.kernel_size**2)
                self.unpools.append(avgunpool)
        
        self.targetneurons = torch.jit.annotate(typing.List[torch.Tensor], [])
        for idx in range(len(self.synapses)+1):
            self.targetneurons.append(torch.tensor([]))

        self.betas = torch.jit.annotate(typing.List[torch.Tensor], [])
        for idx in range(len(self.synapses)+1):
            self.betas.append(torch.tensor([]))

        self.pools = torch.nn.ModuleList(self.pools)
        for idx in range(len(self.pools)):
           self.pools[idx].return_indices = True # we will need the index information to unpool
        #   self.pools[idx] = torch.jit.trace(self.pools[idx], torch.rand(1, self.channels[idx+1], 32, 32))

        self.conv_len = len(self.pools)

        #traceneurons = self.init_neurons(128, torch.device(0))
        #self.betas = self.fieldbeta(0.1)
        #self.energymodel = torch.jit.trace(self.energymodel, (self, self.targetneurons, traceneurons, 10, self.betas, self.fullclamping, torch.nn.CrossEntropyLoss(reduction='none'), False))

    def postupdate(self):
        print()
        #for idx in range(len(self.synapses)):
        #    layer = self.synapses[idx]
        #    transpose = self.syn_transpose[idx]
        #    if isinstance(layer, torch.nn.Conv2d):
        #        transpose.weight = layer.weight.transpose(1,0).flip(2,3)
        #    elif isinstance(layer, torch.nn.Linear):
        #        transpose.weight = layer.weight.T

    def convlayer(self, idx, neurons, grads):
        ## forward flow
        pool : ModuleInterface = self.pools[idx]
        syn : ModuleInterface = self.synapses[idx]
        grads[idx+1] += pool.forward(syn.forward(neurons[idx]))
    def convlayerback(self, idx, neurons, grads):
        ## backward flow
        # unpool : ModuleInterface = torch.nn.Identity()#self.unpools[idx]
        #syn_t : ModuleInterface = self.syn_transpose[idx]
        #unpool.forward
        #grads[idx] += syn_t.forward((neurons[idx+1]))

        # transpose = torch.nn.ConvTranspose2d(layer.out_channels, layer.in_channels, layer.kernel_size, stride=layer.stride,
        layer : ModuleInterface = self.synapses[idx]
        kern0, kern1 = layer.kernel_size
        pad0, pad1 = layer.padding
        # yes, this is the same as a convTranspose, but a normal convolution (with the kernels appropriately tranposed/flipped) uses way less memory.
        #transpose = torch.nn.Conv2d(layer.out_channels, layer.in_channels, (kern1,kern0), stride=layer.dilation, dilation = layer.stride,
        #                        padding=(kern1-1-pad0,kern0-1-pad1), bias=False)
        #weight = layer.weight.transpose(1,0).flip(2,3)
        #grads[idx] += F.conv2d((neurons[idx+1]), weight, stride=layer.dilation, dilation=layer.stride,padding=(kern1-1-pad0,kern0-1-pad1))
        grads[idx] += F.conv_transpose2d((neurons[idx+1]), layer.weight, stride=layer.stride, padding=layer.padding)

    def fclayerforward(self, idx, neurons, grads, mbs):
        syn : ModuleInterface = self.synapses[idx]
        grads[idx+1] += syn.forward(neurons[idx].view(mbs,-1))

    def fclayerbackward(self, idx, neurons, grads):
        #syn_t : ModuleInterface = self.syn_transpose[idx]
        #grads[idx] += syn_t.forward(neurons[idx+1]).view(neurons[idx].size())
        syn : ModuleInterface = self.synapses[idx]
        grads[idx] += F.linear(neurons[idx+1], syn.weight.T).view(neurons[idx].size())

    def latlayer(self, j, idx, neurons, grads, mbs):
        lat_syn : ModuleInterface = self.lat_syn[j]
        grads[idx] += lat_syn.forward(neurons[idx].view(mbs, -1)).view(neurons[idx].size())
        # in this case, the upper/lower triangle function as the forward/backward for a self-connected layer.
        # no need to do backwards flow.
        
    def dPhi(self, grads, targetneurons, neurons, betas, fullclamping, criterion):
        for idx in range(len(grads)):
            grads[idx].zero_()
        mbs = neurons[0].size(0)

        conv_len: int = self.conv_len #len(self.pools)

        futures : List[torch.jit.Future[torch.Tensor]] = []

        # convolutional layers

        for idx in range(conv_len):
            futures.append(torch.jit.fork(self.convlayer, idx, neurons, grads))
        for idx in range(conv_len):
            futures.append(torch.jit.fork(self.convlayerback, idx, neurons, grads))

        # fully-connected layers

        ## forward flow
        for idx in range(conv_len, len(self.synapses)):
            futures.append(torch.jit.fork(self.fclayerforward, idx, neurons, grads, mbs))
        ## backward flow
        for idx in range(conv_len, len(self.synapses)):
            futures.append(torch.jit.fork(self.fclayerbackward, idx, neurons, grads))

        # lateral layers
        for j, idx in enumerate(self.lat_layer_idxs):
            futures.append(torch.jit.fork(self.latlayer, j, idx, neurons, grads, mbs))

        # nudge
        #if targetneurons[0].size(0) > 0:
        #if self.trainreconstruct:
        grads[0] -= 0.5*(betas[0]*(neurons[0] - targetneurons[0])) 
        for idx in range(1,len(targetneurons)):
            # if self.targetneurons[idx].size(0) > 0:
            #if 'MSELoss' in criterion.__class__.__name__:
            grads[idx] -= 0.5 * (betas[idx]*(neurons[idx] - targetneurons[idx]))

        return grads

    def energymodel(self, targetneurons, neurons, T: int, betas, fullclamping, criterion, check_thm=False):
        # T = torch.jit.annotate(int, T)
        not_mse = True #criterion.__class__.__name__.find('MSE')==-1)
        mbs = neurons[0].size(0)
        device = neurons[0].device

        # apply full clamping
        with torch.no_grad():
            for idx in range(len(neurons)):
                # neurons[idx].is_leaf = check_thm
                # if fullclamping[idx].size(0) >  0:
                neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]
                # neurons[idx].requires_grad = True

        grads = []
        for idx in range(len(neurons)):
            grads.append(torch.zeros_like(neurons[idx]))#, requires_grad=check_thm))

        for idx, pool in enumerate(self.pools): #range(len(self.pools)):
           #self.pools[idx]
           pool.return_indices = False #True # we will need the index information to unpool
        #   self.unpooldata[idx].to(device)

        with torch.no_grad():
            # simulate dynamics for T timesteps
            for t in range(T):
                grads = self.dPhi(grads, targetneurons, neurons, betas, fullclamping, criterion)

                #print("")
                #print("================================= TN ===============================")
                #print(targetneurons)
                #print("")
                #print("============================= grads ========================")
                #print(grads)
                #for idx in range(len(grads)):
                #    print('targneur, grads', targetneurons[idx].sum(), grads[idx].sum())

                for idx in range(0,len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    # if check_thm:
                    #   neurons[idx].retain_grad()
                    # else:
                    # neurons[idx].requires_grad = True
             
                if False: #not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )
                # if chek_thm:
                #   neurons[-1].retain_grad()
                # neurons[-1].requires_grad = True

                # apply full clamping
                # with torch.no_grad():
                for idx in range(len(neurons)):
                    # if fullclamping[idx].size(0) >  0:
                    neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]

            for idx in range(len(self.pools)):
                self.pools[idx].return_indices = False # reset value for other functions

            #print('gh', [g.sum() for g in grads])
            #print('nh', [n.sum() for n in neurons])
            return neurons

    def forward(self, x, y, neurons, T, beta=0.0) -> typing.List[torch.Tensor]:#, criterion="ARGUMENT IGNORED! DO NOT USE THIS ARGUMENT", check_thm=False):
        #torch.nn.MSELoss(reduction='none')
        for idx in range(len(neurons)):
            if neurons[idx].is_leaf:
                neurons[idx].requires_grad_(False)

        #neurons = Reversible_CNN.forward(self, x, y, neurons, T, beta=beta, criterion=self.criterion, check_thm=False)#check_thm)
        # not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        self.targetneurons = self.xytotargetneurons(x, y)
        self.betas = self.fieldbeta(beta)

        return self.energymodel(self.targetneurons, neurons, T, self.betas, self.fullclamping, self.criterion, check_thm=False)


        for idx in range(len(neurons)):
            #if neurons[idx].is_leaf:
            neurons[idx].requires_grad_(False)
        
        return neurons

    #def init_neurons(self, mbs, device):
    #    neurons = Reversible_CNN.init_neurons(self, mbs, device)
    #    
    #    for idx in range(len(self.unpools)):
    #        self.unpooldata[idx] = torch.tensor([], device=device)

    #    return neurons


