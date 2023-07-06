import torch
from model_utils import * 



# Multi-Layer Perceptron wtih Lateral-Connections

class Lat_P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(Lat_P_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False    #Softmax readout is only defined for CNN and VFCNN       
        self.nc = self.archi[-1]

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))
        # Lateral synapses (connections between neurons within a layer, not to a different layer)
        self.lat_syn = torch.nn.ModuleList()
        for idx in range(1, len(archi)):
            self.lat_syn.append(torch.nn.Linear(archi[idx], archi[idx], bias=True))

            
    def Phi(self, x, y, neurons, beta, criterion):
        #Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss 
        x = x.view(x.size(0),-1) # flattening the input
        
        layers = [x] + neurons  # concatenate the input to other layers
        
        # Primitive function computation
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.W.s_n-1
        for idx in range(len(self.lat_syn)):
            phi += torch.sum( self.lat_syn[idx](layers[idx+1]) * layers[idx+1], dim=1).squeeze() # essentially a hopfield potential between the neurons within one layer
            # idx+1 to skip input layer
        
        if beta!=0.0: # Nudging the output layer when beta is non zero 
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi -= beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
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
        





# lateral-connectivity, mulit-head CNN
class Lat_MH_CNN(P_CNN):
    def __init__(self, lat_heads, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False):
        # initialize default P_CNN structure, without the final fully connected layer (pass in the last layer to store proper output size)
        super(Lat_MH_CNN, self).__init__(in_size, channels, kernels, strides, [fc[-1]], pools, paddings, activation=activation, softmax=softmax) 
        self.synapses = self.synapses[:-1] # remove final fully connected layer, we will add lateral-connected heads in first
        self.lat_heads = torch.Tensor(lat_heads).to(int)
        
        size = in_size
        
        # layers have already been added by super(), just compute the size
        for idx in range(len(channels)-1): 
#             self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
#                                                  stride=strides[idx], padding=paddings[idx], bias=True))
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]        
        
        
        # split into multiple linear projections, use lateral connections to form attention
        self.head_encoders = torch.nn.ModuleList()
        self.head_hopfield = torch.nn.ModuleList()
        self.head_pos = [0] # torch.cat((torch.Tensor(0), torch.cumsum(self.lat_heads, 0)), 0).to(int)
#         self.head_decodes = torch.nn.ModuleList()
        for idx in range(len(lat_heads)):
            # projects from the last convolutional layer, uses lateral connection, then projects back
            self.head_encoders.append(torch.nn.Linear(size, lat_heads[idx], False))
            self.head_hopfield.append(torch.nn.Linear(lat_heads[idx], lat_heads[idx], bias=False))
            self.head_pos.append(self.head_pos[idx] + self.lat_heads[idx])
            # should head encoder and hopfield have bias?
        size = sum(lat_heads)
        self.head_pos = torch.Tensor(self.head_pos).to(int)
        
        # adapt first fully connected layer to take input from heads
        # fully connect it back down to output dimension
        fc_layers = [size] + fc
        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        
            
    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        heads = len(self.lat_heads)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        #Phi computation changes depending on softmax == True or not
        for idx in range(conv_len):    
            phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()  
        
        for j in range(heads):
            head = layers[conv_len+1][:,self.head_pos[j]:self.head_pos[j+1]]
            phi += torch.sum(self.head_encoders[j](layers[conv_len].view(mbs, -1)) * head, dim=1).squeeze()
            phi += torch.sum(self.head_hopfield[j](head) * head, dim=1).squeeze()
            
        if not self.softmax:
            for idx in range(conv_len, tot_len):
                layeridx = idx + 1 # head encoders and lateral synapses not in synapses[]  but head neurons are in layers[]
                phi += torch.sum( self.synapses[idx](layers[layeridx].view(mbs,-1)) * layers[layeridx+1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi -= beta*L

        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len, tot_len-1):
                layeridx = idx + heads
                phi += torch.sum( self.synapses[idx](layers[layeridx].view(mbs,-1)) * layers[layeridx+1], dim=1).squeeze()
             
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                phi -= beta*L            
        
        return phi
    
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
      
        # for j in range(len(self.lat_heads)):
        neurons.append(torch.zeros((mbs, int(torch.sum(self.lat_heads).data)), requires_grad=True, device=device))
      
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
        
        # force hopfield / lateral connections to be symmetric (backwards and forward weights the same)
        for j in range(len(self.lat_heads)):
            with torch.no_grad():
                self.head_hopfield[j].weight = torch.nn.Parameter(0.5 * (self.head_hopfield[j].weight + self.head_hopfield[j].weight.T))
                # unlike the inter-layer connections, not each weight represents a unique pair of neurons.
                # the upper and lower triangle of this square matrix are the backwards and forwards connections
                # these need to be the same here.

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
        

# lateral-connectivity on logit/classification output layer to produce softmax-like behaviour CNN
class fake_softmax_CNN(P_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False):
        # initialize default P_CNN structure, without the final fully connected layer (pass in the last layer to store proper output size)
        softmax = false
        
        super(Lat_MH_CNN, self).__init__(in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax) 
        
        self.lat_syn = torch.nn.Linear(fc[-1], fc[-1], bias=False) 

    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        heads = len(self.lat_heads)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        #Phi computation changes depending on softmax == True or not
        for idx in range(conv_len):    
            phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()  
        
           
        if not self.softmax:
            for idx in range(conv_len, tot_len):
                layeridx = idx + 1 # head encoders and lateral synapses not in synapses[]  but head neurons are in layers[]
                phi += torch.sum( self.synapses[idx](layers[layeridx].view(mbs,-1)) * layers[layeridx+1], dim=1).squeeze()
            
            # apply competitive lateral connection to readout classification layer for softmax-like behaviour
            phi += torch.sum( self.lat_syn(layers[-1]) * layers[-1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi -= beta*L

        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len, tot_len-1):
                layeridx = idx + heads
                phi += torch.sum( self.synapses[idx](layers[layeridx].view(mbs,-1)) * layers[layeridx+1], dim=1).squeeze()
             
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                phi -= beta*L            
        
        return phi
    
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
      
        # for j in range(len(self.lat_heads)):
        # neurons.append(torch.zeros((mbs, int(torch.sum(self.lat_heads).data)), requires_grad=True, device=device))
      
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
        
        # force lateral connections to be symmetric (backwards and forward weights the same)
        # and set them based on the inputs to the last layer, so they are always more inhibitory than the input can overcome
        with torch.no_grad():
            inhibitstrength = self.synapses[-1].weight.sum()/10 # amount a given neuron would be stimulated if previous layer was fully active
            self.lat_syn.weight = torch.nn.Parameter(torch.full(self.lat_syn.weight.size(), -inhibitstrength)
            self.lat_syn.weight = self.lat_syn.weight + inhibitstrength*torch.eye(self.lat_syn.weight.size()[1]) # remove self-connections
            # self.head_hopfield[j].weight = torch.nn.Parameter(0.5 * (self.head_hopfield[j].weight + self.head_hopfield[j].weight.T))


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
        
