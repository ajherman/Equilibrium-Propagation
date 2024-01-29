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
            self.lat_syn.append(torch.nn.Linear(archi[idx], archi[idx], bias=False))

            
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
        # force hopfield / lateral connections to be symmetric (backwards and forward weights the same)
        for j in range(len(self.lat_syn)):
            with torch.no_grad():
                self.lat_syn[j].weight = torch.nn.Parameter(0.5 * (self.lat_syn[j].weight + self.lat_syn[j].weight.T) - torch.diag(torch.diagonal(self.lat_syn[j].weight)) )
                # unlike the inter-layer connections, not each weight represents a unique pair of neurons.
                # the upper and lower triangle of this square matrix are the backwards and forwards connections
                # these need to be the same here.
                # also removing the diagonal (neuron self connections). But maybe this would be valid to keep?

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
        

class lat_CNN(P_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=hard_sigmoid, softmax=False):
        # initialize default P_CNN structure        
        P_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax) 

        self.lat_constraints = lat_constraints
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
            if idx in self.lat_layer_idxs:
                fcsize = size * size * channels[idx+1]
                self.lat_syn.append(torch.nn.Linear(fcsize, fcsize, bias=False))

        size = size * size * channels[-1]        
        layeridx = len(channels)-1
        # fully connect it back down to output dimension
        fc_layers = [size] + fc
        for idx in range(len(fc)):
            # self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
            if layeridx + idx in self.lat_layer_idxs:
                self.lat_syn.append(torch.nn.Linear(fc_layers[idx+1], fc_layers[idx+1], bias=False))

    def postupdate(self):
        with torch.no_grad():
            for i, constraint in enumerate(self.lat_constraints):
                if 'zerodiag' in constraint:
                    # zero diagonal to remove self-interaction
                    self.lat_syn[i].weight.data -= torch.diag(torch.diag(self.lat_syn[i].weight.data))
                if 'transposesymmetric' in constraint:
                    self.lat_syn[i].weight.data = 0.5*(self.lat_syn[i].weight.data + self.lat_syn[i].weight.data.transpose(0,1))
                if 'negReLu' in constraint:
                    self.lat_syn[i].weight.data = -F.relu(-self.lat_syn[i].weight.data)


    def setmode(train_vert=True, train_lat=True):
        for idx in range(len(self.synapses)):
            self.synapses[idx].requires_grad_(trian_vert)
        for idx in range(len(self.lat_syn)):
            self.lat_syn[idx].requires_grad_(train_lat)

    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        #Phi computation changes depending on softmax == True or not
        for idx in range(conv_len):    
            phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()  
        
        # apply lateral connections to layer
        for i, lidx in enumerate(self.lat_layer_idxs):
            idx = lidx + 1
            phi += torch.sum( self.lat_syn[i](layers[idx].view(mbs,-1)) * layers[idx].view(mbs,-1), dim=1 ).squeeze()

        if not self.softmax:
            for idx in range(conv_len, tot_len):
                phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()

            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(F.softmax(layers[-1].float(), dim=1), y).squeeze()             
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
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
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
                grads = torch.autograd.grad(phi, neurons, grad_outputs=self.init_grads, create_graph=False)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].requires_grad = True
             
                if False: #not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
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
      
        # for j in range(len(self.lat_heads)):
        # neurons.append(torch.zeros((mbs, int(torch.sum(self.lat_heads).data)), requires_grad=True, device=device))
      
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            

        self.init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
          
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
        




# lateral interactions with convolution rather than all-to-all, at least in layers with convolutional input
class lat_conv_CNN(lat_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_kernels, lat_layer_idxs, lat_constraints, activation=hard_sigmoid, softmax=False):
        
        for i in range(len(lat_layer_idxs)):
            while lat_layer_idxs[i] < 0:
                lat_layer_idxs[i] += len(kernels) + len(fc)

        fc_lat_idxs = list(filter(lambda x: x >= len(kernels), lat_layer_idxs))
        conv_lat_idxs = list(filter(lambda x: x < len(kernels), lat_layer_idxs))

        print('fc lat layers:', fc_lat_idxs)
        print('conv lat layers:', conv_lat_idxs)

        # initialize lat_CNN structure for fully connected layers
        lat_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, fc_lat_idxs, lat_constraints, activation=activation, softmax=softmax) 

        self.conv_lat_layer_idxs = conv_lat_idxs
        
        self.conv_lat_syn = torch.nn.ModuleList()

        j = 0
        for idx in range(len(kernels)):
            if idx in self.conv_lat_layer_idxs:
                lat_kern_size = lat_kernels[j]
                assert lat_kern_size % 2 == 1, "lateral convolutional kernel must have odd size! (self-to-self, must have center element to be aligned)"
                pad = int( ( lat_kern_size - 1 ) / 2 )
                self.conv_lat_syn.append(torch.nn.Conv2d(channels[idx+1], channels[idx+1], lat_kern_size, padding=pad, bias=False))
                j += 1

    def postupdate(self):
        with torch.no_grad():
            for i in range(len(self.conv_lat_syn)):
                constraint = self.lat_constraints[i]
                if 'zerodiag' in constraint:
                    # remove self-connections (same feature to same feature to the same pixel -> [rowidx, rowidx, centerx, centery])
                    centerx = math.floor(self.conv_lat_syn[i].kernel_size[0]/2)
                    centery = math.floor(self.conv_lat_syn[i].kernel_size[1]/2)
                    self.conv_lat_syn[i].weight.data[:,:,centerx,centery] -= torch.diag(torch.diag(self.conv_lat_syn[i].weight.data[:,:,centerx,centery]))
                if 'transposesymmetric' in constraint:
                    self.conv_lat_syn[i].weight.data = 0.5*(self.conv_lat_syn[i].weight.data + self.conv_lat_syn[i].weight.data.transpose(0,1))
                if 'negReLu' in constraint:
                    self.conv_lat_syn[i].weight.data = -F.relu(-self.conv_lat_syn[i].weight.data)
                

            for i in range(len(self.lat_syn)):
                constraint = self.lat_constraints[i+len(self.conv_lat_syn)]
                if 'zerodiag' in constraint:
                    # zero diagonal to remove self-interaction
                    self.lat_syn[i].weight.data -= torch.diag(torch.diag(self.lat_syn[i].weight.data))
                if 'transposesymmetric' in constraint:
                    self.lat_syn[i].weight.data = 0.5*(self.lat_syn[i].weight.data + self.lat_syn[i].weight.data.transpose(0,1))
                if 'negReLu' in constraint:
                    self.lat_syn[i].weight.data = -F.relu(-self.lat_syn[i].weight.data)

    def Phi(self, x, y, neurons, beta, criterion=torch.nn.MSELoss(reduction='none')):
        phi = lat_CNN.Phi(self, x, y, neurons, beta, criterion)

        # apply convolutional lateral connections to layer
        for i, idx in enumerate(self.conv_lat_layer_idxs):
            phi += torch.sum( self.conv_lat_syn[i](neurons[idx]) * neurons[idx] , dim=(1,2,3) ).squeeze()

        return phi



# lateral inhibition locally in convolutional layer to encourage sparsity, robustness
class LCACNN(lat_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, inhibitstrength, competitiontype, lat_layer_idxs, lat_constraints, sparse_layer_idxs=[-1], comp_syn_constraints=['zerodiag+transposesymmetric'], activation=hard_sigmoid, softmax=False, layerlambdas=[1.0e-2], dt=1.0):
        self.layerlambdas = layerlambdas
        self.inhibitstrength = inhibitstrength*0.5 # used in an energy function rather than EOM, need 1/2 coefficient (vector is squared). derivative then is 1*feature inner products
        self.competitiontype = competitiontype
        lat_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=activation, softmax=softmax)

        self.dt = dt
        print('timestep:', self.dt)

        for idx in range(len(sparse_layer_idxs)):
            if sparse_layer_idxs[idx] < 0:
                sparse_layer_idxs[idx] += len(self.synapses)
        self.sparse_layer_idxs = sparse_layer_idxs
        self.comp_syn_constraints = comp_syn_constraints

        self.conv_comp_layers = torch.nn.ModuleList()
        self.fc_comp_layers = torch.nn.ModuleList()
        size = in_size

        # convolutional (local) competition: kernel connects all neurons which share any of their input
        for idx in range(len(kernels)):
            if idx in sparse_layer_idxs:
                compete_radius = math.ceil((kernels[idx])/(strides[idx])) - 1 # number of neurons in one direction between this neuron and the first neuron without any of the same inputs (exclusive)
                compete_range = compete_radius*2 + 1 # width of block of neurons which share any input neurons
                layer = torch.nn.Conv2d(channels[idx+1], channels[idx+1], compete_range, padding=compete_radius, bias=False)
                layer.requires_grad_(False)
                self.conv_comp_layers.append(layer)
                #self.conv_comp_layers[idx].requires_grad_(False)

        for idx in range(len(fc)):
            if idx+len(kernels) in sparse_layer_idxs:
                layer = torch.nn.Linear(fc[idx], fc[idx], bias=False)
                layer.requires_grad_(False)
                self.fc_comp_layers.append(layer)
                #self.fc_comp_layers[idx].requires_grad_(False)

        self.setlat()
    
   # def setmode(train_vert, train_lat, train_comp):
   #     RevLatCNN.setmode(train_vert, train_lat)
   # 
   #     for idx in range(len(self.conv_comp_layers)):
   #         self.conv_comp_layers[idx].requires_grad_(train_comp)
   #     for idx in range(len(self.fc_comp_layers)):
   #         self.fc_comp_layers[idx].requires_grad_(train_comp)

    def setlat(self):

        with torch.no_grad():
            # initialize competition lateral connections
            if self.competitiontype == 'feature_inner_products':
                # LCA-like sparse coding lateral inhibition based on inner product of features (rows of weight of last layer)
                for j, layer in enumerate(self.conv_comp_layers):
                    idx = self.sparse_layer_idxs[j]
                    features = self.synapses[idx].weight.data #/ self.synapses[idx].weight.data.norm(2, dim=(1,2,3))[:,None,None,None]
                    for rowidx in range(features.size(0)):
                        layer.weight[rowidx,:,:,:] = (- self.inhibitstrength * (features * features[rowidx,:,:,:]).sum(dim=(1,2,3)))[None,:,None,None].expand(1, -1, layer.kernel_size[0], layer.kernel_size[1])
                        # remove self-connections (same feature to same feature to the same pixel -> [rowidx, rowidx, centerx, centery])
                        #centerx = math.floor(layer.kernel_size[0]/2)
                        #centery = math.floor(layer.kernel_size[1]/2)
                        #layer.weight[rowidx,rowidx,centerx,centery].zero_()
                        #layer.bias.zero_()
                        #layer.bias.fill_(-self.layerlambdas[j])

                conv_comp_len = j+1
                for j, layer in enumerate(self.fc_comp_layers):
                    idx = self.sparse_layer_idxs[j+conv_comp_len]
                    features = self.synapses[idx].weight.data #/ self.synapses[idx].weight.data.norm(2, dim=1)[:,None]
                    for rowidx in range(features.size(0)):
                        layer.weight[:,rowidx] = - self.inhibitstrength * (features * features[rowidx,:]).sum(dim=1)
                        #layer.weight[rowidx,rowidx].zero_()
                        #layer.bias.zero_()
                        #layer.bias.fill_(-self.layerlambdas[j+conv_comp_len])
                    layer.weight.data = layer.weight.data.contiguous() # setting columns with rows makes it column rather than row major in memory
            elif self.competitiontype == 'uniform_inhibition':
                for layer in self.conv_comp_layers:
                    layer.weight = layer.weight.fill_(-self.inhibitstrength)
                for layer in self.fc_comp_layers:
                    layer.weight = layer.weight.fill_(-self.inhibitstrength)
            else:
                print('UNKNOW VALUE {} for competition_type!!'.format(self.competitiontype))

    def postupdate(self, setlat=True):
        lat_CNN.postupdate(self)
        
        with torch.no_grad():
            for i, constraint in enumerate(self.comp_syn_constraints):
            #for j, idx in enumerate(self.sparse_layer_idxs):
            #    # cosntrain the input features to sparse coding layers to norm 1
                idx = self.sparse_layer_idxs[i]
                if 'rowunitnorm' in constraint:
                    if isinstance(self.synapses[idx], torch.nn.Conv2d):
                        self.synapses[idx].weight /= self.synapses[idx].weight.norm(2, dim=(1,2,3))[:,None,None,None]
                    elif isinstance(self.synapses[idx], torch.nn.Linear):
                        self.synapses[idx].weight /= self.synapses[idx].weight.norm(2, dim=1)[:,None]
                if 'colunitnorm' in constraint:
                    if isinstance(self.synapses[idx], torch.nn.Conv2d):
                        self.synapses[idx].weight /= self.synapses[idx].weight.norm(2, dim=(0,2,3))[None,:,None,None]
                    elif isinstance(self.synapses[idx], torch.nn.Linear):
                        self.synapses[idx].weight /= self.synapses[idx].weight.norm(2, dim=0)[None,:]
                if 'fixedlambda' in constraint:
                    #self.conv_comp_layers[i].bias.fill_(-self.layerlambdas[i])
                    #self.synapses[idx].bias.fill_(-self.layerlambdas[i])
                    pass # lambda sets threshold for membrane potential, don't include in bias
                
                #self.synapses[idx].bias.fill_(-self.layerlambdas[j])
        if setlat:
            self.setlat()
        
        with torch.no_grad():
            for i, constraint in enumerate(self.comp_syn_constraints):
                if i < len(self.conv_comp_layers):
                    layer = self.conv_comp_layers[i]
                    if 'zerodiag' in constraint:
                        # zero diagonal to remove self-interaction
                        centerx = math.floor(layer.kernel_size[0]/2)
                        centery = math.floor(layer.kernel_size[1]/2)
                        layer.weight[:,:,centerx,centery] -= torch.diag(torch.diag(layer.weight[:,:,centerx,centery]))
                else:
                    layer = self.fc_comp_layers[i-len(self.conv_comp_layers)]
                    if 'zerodiag' in constraint:
                        layer.weight -= torch.diag(torch.diag(layer.weight))
                if 'transposesymmetric' in constraint:
                    layer.weight.data = 0.5*(layer.weight.data + layer.weight.data.transpose(0,1)) # order is important, otherwise it will be transposed in memory
                if 'negrelu' in constraint:
                    layer.weight.data = -F.relu(-layer.weight.data)



    def Phi(self, x, y, neurons, beta, criterion):
        phi = lat_CNN.Phi(self, x, y, neurons, beta, criterion)
        
        conv_len = len(self.channels)-1

        for j, layer in enumerate(self.conv_comp_layers):
            idx = self.sparse_layer_idxs[j]# + 1
            phi += torch.sum(layer(neurons[idx]) * neurons[idx], dim=(1,2,3))
            # obsolete the L2 term on this layer, use L1 instead (from the constant negative bias =lambda)
            # this also prevents step-size 1 from instantly decaying the current value to zero.
            #phi += 0.5* neurons[idx].norm(2, dim=(1,2,3))

        for j, layer in enumerate(self.fc_comp_layers):
            idx = self.sparse_layer_idxs[j+len(self.conv_comp_layers)]# + 1
            phi += torch.sum(layer(neurons[idx]) * neurons[idx], dim=1)
            #phi += 0.5* neurons[idx].norm(2, dim=(1))

        return phi

    def dPhi(self, grads, x, y, neurons, beta, criterion):
        grads = lat_CNN.dPhi(self, grads, x, y, neurons, beta, criterion)
        
        # lateral layers, no transpose/backward needed, already included in bottom triangle of matrix
        for j, layer in enumerate(self.conv_comp_layers):
            idx = self.sparse_layer_idxs[j]
            grads[idx] += layer(neurons[idx])
        
        for j, layer in enumerate(self.fc_comp_layers):
            idx = self.sparse_layers_idxs[j+len(self.conv_comp_layers)]
            grads[idx] += layer(neurons[idx])

        return grads
    #def compute_syn_grads_sparsity(self, x, y, neurons_1, neurons_2, beta, lambdas, criterion, check_thm=False):
    #    #self.setmode(train_vert=False, train_lat=False, trian_comp=True)
    #    
    #    self.zero_grad()            # p.grad is zero
    #    phi_1 = self.Phi(x, y, neurons_1, beta, criterion)
    #    phi_1 = phi_1.mean()
    #    
    #    phi_2 = self.Phi(x, y, neurons_2, beta, criterion)
    #    phi_2 = phi_2.mean()
    #    
    #    delta_phi = (phi_2 - phi_1)/(lambdas[1] - lambdas[0])
    #    delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        grads = self.init_neurons(mbs, device) # neurons-like shaped arrays for grads
        
        #if check_thm:
        #    for t in range(T):
        #        phi = self.Phi(x, y, neurons, beta, criterion)
        #        init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
        #        grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)

        #        for idx in range(len(neurons)-1):
        #            neurons[idx].retain_grad()
        #     
        #        if not_mse and not(self.softmax):
        #            neurons[-1] = grads[-1]
        #        else:
        #            neurons[-1] = self.activation( grads[-1] )

        #        neurons[-1].retain_grad()
        #else:
        #[n.requires_grad_(False) for n in neurons if n.is_leaf]
        #[g.requires_grad_(False) for g in grads if g.is_leaf]
        
        for t in range(T):
            [n.requires_grad_(True) for n in neurons]
            phi = self.Phi(x, y, neurons, beta, criterion)
            #grads = torch.autograd.grad(phi, neurons, grad_outputs=self.init_grads, create_graph=False)
            phi.sum().backward()
            #[g.zero_() for g in grads]
            #grads = self.dPhi(grads, x, y, neurons, beta, criterion)

            for j, idx in enumerate(self.sparse_layer_idxs):
                self.membranepotentials[j].mul_(1-self.dt) # L2 decay on membrane potential
                self.membranepotentials[j].add_(self.dt*(neurons[idx].grad - neurons[idx])) # integrate neuron input, and L2 penalty on repre.
                neurons[idx] = F.relu(self.membranepotentials[j] - self.layerlambdas[j])

            for idx in range(len(neurons)): # -1):
                if idx not in self.sparse_layer_idxs:
                    neurons[idx].mul_(1-self.dt) # L2 decay 
                    neurons[idx].add_(self.dt*neurons[idx].grad)
                #neurons[idx] = self.activation( (1-self.dt)*neurons[idx] + self.dt*grads[idx] )
                #neurons[idx] = self.activation( grads[idx] )
                #neurons[idx].requires_grad = True

         
            #if not_mse and not(self.softmax):
            #    neurons[-1] = grads[-1]
            #else:
            #    neurons[-1] = self.activation( (1-self.dt)*neurons[-1] + self.dt*grads[-1] )
            #    #neurons[-1] = self.activation( grads[-1] )

            #neurons[-1].requires_grad = True

        return neurons
    
    def init_neurons(self, mbs, device):
        neurons = lat_CNN.init_neurons(self, mbs, device)

        self.membranepotentials = list([torch.zeros_like(neurons[idx]) for idx in self.sparse_layer_idxs])
    
        return neurons


class autoLCACNN(P_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False, dt=0.01, sparse_layer_idxs=[0], lambdas=[0.1], inhibitstrength=1.0):
        P_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax)
        self.origdt = dt
        self.layerlambdas = lambdas
        self.sparse_layer_idxs = sparse_layer_idxs
        self.inhibitstrength = inhibitstrength

    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        for idx in self.sparse_layer_idxs:
            #creates a lot of problems when the grad is on the membpotes instead of the neurons.
            # # layer no longer simply has hopfield pairwise product energy, but minimizes L2 of reconstruction error
            # # mathematically equivelant to adding lateral interactions = - 1/2 feature inner products, but faster to do with autograd rather than another convolution
            # # layers[idx]- could be replaced by a seperate dot product term like other layers for input drive. Faster to do competition and drive together in one term though.
            # if idx-1 in self.sparse_layer_idxs: # for stacked LCA, membrane potential is fed to next layer to reconstruct, not actual activations
            #     target = self.membpotes[idx-1]
            #     target.grad = None
            # else:
            target = layers[idx]
            if isinstance(self.synapses[idx], torch.nn.Conv2d):
                if idx < 1:
                    p = torch.nn.Identity()
                else:
                    p = self.pools[idx-1]
                #phi -= (p(target)-F.conv_transpose2d(layers[idx+1], self.synapses[idx].weight, padding=self.synapses[idx].padding, stride=self.synapses[idx].stride, bias=None)).pow(2).sum(dim=(1,2,3))*0.5
                phi -= self.inhibitstrength*(F.conv_transpose2d(layers[idx+1], self.synapses[idx].weight, padding=self.synapses[idx].padding, stride=self.synapses[idx].stride, bias=None)).pow(2).sum(dim=(1,2,3))*0.5
            elif isinstance(self.synapses[idx], torch.nn.Linear):
                if isinstance(self.synapses[idx-1], torch.nn.Conv2d):
                    p = self.pools[idx-1]
                else:
                    p = torch.nn.Identity()
                phi -= (p(target).view(mbs,-1)-torch.matmul(layers[idx+1], self.synapses[idx].weight)).pow(2).sum(dim=1)*0.5

        # remaning layers are simple hopfield energy
        for idx in range(conv_len):
            if idx in self.sparse_layer_idxs:
                # pool representation, since we don't apply the pool to its input already so all values are part of state not only max 
                p = torch.nn.Identity()
            else:
                p = self.pools[idx]

            if idx-1 in self.sparse_layer_idxs:
                # prevent feedback to sparse layer from hopfield layers with .detach() to hide gradient
                #.detach()
                phi += torch.sum( p(self.synapses[idx](self.pools[idx-1](layers[idx]))) * layers[idx+1], dim=(1,2,3)).squeeze()     
            else:
                phi += torch.sum( p(self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     

        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len, tot_len):
                if idx not in self.sparse_layer_idxs:
                    if idx-1 in self.sparse_layer_idxs:
                        phi += torch.sum( self.synapses[idx](self.pools[idx-1](layers[idx]).view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                    else:
                        phi += torch.sum( self.synapses[idx]((layers[idx]).view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                     
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi -= beta*L
        else:
            for idx in range(conv_len, tot_len-1):
                if idx not in self.sparse_layer_idxs:
                    if idx-1 in self.sparse_layer_idxs:
                        phi += torch.sum( self.synapses[idx](self.pools[idx-1](layers[idx]).view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                    else:
                        phi += torch.sum( self.synapses[idx]((layers[idx]).view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                if idx-1 in self.sparse_layer_idxs:
                    out = self.synapses[-1](self.pools[-2](layers[-1]).view(mbs,-1))
                else:
                    out = self.synapses[-1](layers[-1].view(mbs,-1))
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(out.float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(out.float(), y).squeeze()  
                phi -= beta*L            

        return phi


    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     

        for s in self.synapses:
            s.requires_grad_(False) # for ep, we dont' need gradients on the weights even during training evaluation, since we calculate this with the theorem.
        
        #if check_thm:
        #    for t in range(T):
        #        phi = self.Phi(x, y, neurons, beta, criterion)
        #        init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
        #        grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)

        #        for idx in range(len(neurons)-1):
        #            neurons[idx] = self.activation( grads[idx] )
        #            neurons[idx].retain_grad()
        #     
        #        if not_mse and not(self.softmax):
        #            neurons[-1] = grads[-1]
        #        else:
        #            neurons[-1] = self.activation( grads[-1] )

        #        neurons[-1].retain_grad()
        #else:
        auxdt = 1.0 # non-lca model dynamics need more time to accomplish anything
        for t in range(T):

            phi = self.Phi(x, y, neurons, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm)
             
            #print('neurons grad before', [type(n.grad) for n in neurons])
            #print('neurons grad before', [(n.requires_grad) for n in neurons])
            #print('neurons leaf before', [(n.is_leaf) for n in neurons])
            #print('membpot grad before', [type(m.grad) for m in self.membpotes])
            #phi.sum().backward(retain_graph = check_thm)
            #print('neurons grad after ', [type(n.grad) for n in neurons])
            #print('neurons grad after ', [(n.requires_grad) for n in neurons])
            #print('neurons leaf after ', [(n.is_leaf) for n in neurons])
            #print('membpot grad after ', [type(m.grad) for m in self.membpotes])

            ##self.dt = mbs*1/((neurons[0].grad-neurons[0]).norm(2)+5*mbs) # timestep adapts to make each step the same size, with maximum of 0.1
            #with torch.no_grad(): # must turn off grad so the computation graph doesn't get too smart and go backwards to past iterations using membpotes
            # integrate representation  layer gradient into non-thresholded membrane potential, then view this with ReLu
            for idx in range(len(neurons)):
                if idx in self.sparse_layer_idxs:
                    if idx < len(neurons)-1: # don't apply L2 decay on membrane potential if there isn't a layer above
                        #self.membpotes[idx].mul_(1-self.dt)
                        self.membpotes[idx] = (1-self.dt)*self.membpotes[idx].detach() + self.dt*grads[idx] + self.dt*neurons[idx].detach()
                    else:
                        self.membpotes[idx] = self.membpotes[idx].detach() + self.dt*grads[idx]
                    #if self.membpotes[idx].is_leaf:
                    #    self.membpotes[idx].requires_grad = True

                    #self.membpotes[idx].add_(self.dt*(grads[idx]))#- neurons[0])#
                    #if self.membpotes[idx].grad is not None:
                    #    self.membpotes[idx].add_(self.dt*self.membpotes[idx].grad)
                    neurons[idx] = F.relu(self.membpotes[idx] - self.layerlambdas[idx])
                    #neurons[idx] = F.relu(neurons[idx].detach() + grads[idx] - self.layerlambdas[idx])
                    
                    # check removing effect of membrane potentials
                    #self.membpotes[idx] = F.relu(self.membpotes[idx])
                else:
                    # remaining layers do classic hopfield dynamics
                    #neurons[idx].mul_(1-auxdt)
                    #neurons[idx].add_(auxdt*grads[idx])
                    neurons[idx] = self.activation( (1-auxdt)*neurons[idx].detach() + auxdt*grads[idx] )
                #if neurons[idx].is_leaf:
                if not check_thm:
                    neurons[idx].requires_grad = True
                else:
                    neurons[idx].retain_grad()

            #maxdt = 2*self.origdt
            #if self.dt < maxdt:
            #    self.dt += (self.origdt)/200#timestepadapt
            #    if self.dt > maxdt:
            #        self.dt = maxdt
         
            #    neurons[-1] = grads[-1]
            #else:
            #    neurons[-1] = self.activation( grads[-1] )

            #neurons[-1].requires_grad = True

        return neurons

        
    def init_neurons(self, mbs, device):
        #neurons = P_CNN.init_neurons(self, mbs, device)
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if idx in self.sparse_layer_idxs:
                append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))
                if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                    size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            else:
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
            

        # membrane potential varible for sparse coding layer
        self.membpotes = copy(neurons)#[torch.zeros_like(neurons[idx]) for idx in self.sparse_layer_idxs]

        self.dt = self.origdt # adaptive timestep starts small and changes over evolution

        self.stateT = 0

        return neurons

    def Hopfield(self, x, y, neurons, beta, criterion):
        # the plain hopfield energy, for use by differeniating to get STDP-like pre-post activity products
        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        for idx in range(0,conv_len):
            # pool comes after for sparse layers, not before
            # pool representation, since we don't apply the pool to its input already so all vlaues are part of state not only max 
            if idx-1 in self.sparse_layer_idxs:
                p1 = self.pools[idx-1]
            else:
                p1 = torch.nn.Identity()
            if idx in self.sparse_layer_idxs:
                p0 = torch.nn.Identity()
            else:
                p0 = self.pools[idx]
            phi += torch.sum( p0(self.synapses[idx](p1(layers[idx]))) * layers[idx+1], dim=(1,2,3)).squeeze()     
        for idx in range(conv_len, tot_len-1):
            if idx-1 in self.sparse_layer_idxs:
                p1 = self.pools[idx-1]
            else:
                p1 = torch.nn.Identity()
            phi += torch.sum( self.synapses[idx](p1(layers[idx]).view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            idx = tot_len-1
            if idx-1 in self.sparse_layer_idxs:
                p1 = self.pools[idx-1]
            else:
                p1 = torch.nn.Identity()
            phi += torch.sum( self.synapses[idx](p1(layers[idx]).view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi -= beta*L
        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                if idx-1 in self.sparse_layer_idxs:
                    out = self.synapses[-1](self.pools[-2](layers[-1]).view(mbs,-1))
                else:
                    out = self.synapses[-1](layers[-1].view(mbs,-1))
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(out.float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(out.float(), y).squeeze()  
                phi -= beta*L            

        return phi

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        for idx in range(len(self.synapses)):
            self.synapses[idx].weight.requires_grad_(True)
        
        #energy = self.Phi
        energy = self.Hopfield
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = energy(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = energy(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = energy(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

#        for idx in range(len(self.synapses)):
#            self.synapses[idx].weight.requires_grad_(False)

    def postupdate(self):
        with torch.no_grad():
            for j in self.sparse_layer_idxs:
                # normalize columns
                if isinstance(self.synapses[j], torch.nn.Conv2d):
                    #self.synapses[j].weight.div_(self.synapses[j].weight.norm(2, dim=(0,2,3))[None,:,None,None] + 1e-6)
                    self.synapses[j].weight.div_(self.synapses[j].weight.norm(2, dim=(1,2,3))[:,None,None,None] + 1e-6)
                elif isinstance(self.synapses[j], torch.nn.Linear):
                    self.synapses[j].weight.div_(self.synapses[j].weight.norm(2, dim=1)[:,None] + 1e-6)
                self.synapses[j].bias.zero_()




# lateral-connectivity on logit/classification output layer to produce softmax-like behaviour CNN
class fake_softmax_CNN(lat_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, inhibitstrength, competitiontype, lat_constraints, activation=hard_sigmoid, softmax=False):
        lat_layer_idxs = [-1] # lateral connections in final layer only
        softmax=False
        self.inhibitstrength = inhibitstrength 
        self.competitiontype = competitiontype
        lat_CNN.__init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, lat_constraints, activation=activation, softmax=softmax)
        for l in self.lat_syn:
            l.weight.requires_grad = False
            l.bias.requires_grad = False
            l.bias.data = l.bias.data.zero_()
        

    def postupdate(self):
        if self.competitiontype == 'feature_inner_products':
            features = self.synapses[-1].weight.data
            for rowidx in range(self.nc):
                self.lat_syn[-1].weight[rowidx,:] = - self.inhibitstrength * (features * features[rowidx,:]).sum(dim=1)
        elif self.competitiontype == 'uniform_inhibition':
            self.lat_syn[-1].weight.fill_(-self.inhibitstrength)
        else:
            print('UNKNOW VALUE {} for competition_type!!'.format(self.competitiontype))
        self.lat_syn[-1].bias.data = self.lat_syn[-1].bias.data.zero_()

        lat_CNN.postupdate(self)






# Convolutional Neural Network, but with SGD rather than just gradient descent to avoid local minima

class P_CNN_Stochastic(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False, dynamicsmomentum=0.0):
        # super(P_CNN, self).__init__()
        if not hasattr(self, 'call_super_init') or self.call_super_init:
            super(P_CNN_Stochastic, self).__init__()


        self.dynamicsmomentum = dynamicsmomentum

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
        
        #self.dynamicsopt = torch.optim.SGD([], momentum=self.dynamicsmomentum)


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
    

    def forward(self, x, y, neurons, T, beta=0.0, check_thm=False, criterion=torch.nn.MSELoss(reduction='none')):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        dynopt_params = []
        for idx in range(len(neurons)):
            dynopt_params.append({'params': neurons[idx], 'lr': 5.0e-1, 'weight_decay': 5.0e-1}) # weight decay to mimc exponential decay (neurons L2 energy term)
        self.dynamicsopt = torch.optim.Adam(dynopt_params)#, momentum=self.dynamicsmomentum)
        
        #if check_thm:
        for t in range(T):
            self.dynamicsopt.zero_grad()
            phi = self.Phi(x, y, neurons, beta, criterion)
            # init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            # grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)
            ((-phi).sum()).backward()
            #print(phi.sum().item(), [n.sum().item() for n in neurons])
            #print('grads', [n.grad.sum().item() for n in neurons])
            self.dynamicsopt.step()
            #print(phi.sum().item(), [n.sum().item() for n in neurons])

            with torch.no_grad():
                for idx in range(len(neurons)-1):
                    neurons[idx].data = self.activation(neurons[idx])
                    #neurons[idx] = self.activation( neurons[idx] )#.detach().requires_grad_()
                    #neurons[idx].retain_grad()
             
                if not_mse and not(self.softmax):
                    pass
                    #neurons[-1] = neurons[-1]#.detach().requires_grad_()
                else:
                    neurons[-1].data = self.activation(neurons[-1])
                    #neurons[-1] = self.activation( neurons[-1] )#.detach().requires_grad_()
            #neurons[-1].grad = None

        #        neurons[-1].retain_grad()
        #else:
        #     for t in range(T):
        #        phi = self.Phi(x, y, neurons, beta, criterion)
        #        init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
        #        grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

        #        for idx in range(len(neurons)-1):
        #            neurons[idx] = self.activation( grads[idx] )
        #            neurons[idx].requires_grad = True
        #     
        #        if not_mse and not(self.softmax):
        #            neurons[-1] = grads[-1]
        #        else:
        #            neurons[-1] = self.activation( grads[-1] )

        #        neurons[-1].requires_grad = True

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
 
