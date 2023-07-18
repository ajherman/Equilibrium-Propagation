import torch
from model_utils import * 

class Reversible_CNN(P_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=my_sigmoid, softmax=False):
        super(Reversible_CNN, self).__init__(in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=softmax)

        self.fullclamping = []
        for idx in range(len(self.synapses)+1):
            self.fullclamping.append(torch.tensor([]))

        self.mode(trainclassifier=True, trainreconstruct=False)

        self.reconcriterion = torch.nn.MSELoss(reduction = 'none')

    def mode(self, trainclassifier=True, trainreconstruct=False):
        self.trainclassifier = trainclassifier
        self.trainreconstruct = trainreconstruct

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
            phi -= 0.5*(betas[0]*self.reconcriterion(neurons[0], targetneurons[0])).view(mbs,-1).sum(dim=1).squeeze() 
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
            self.betas[0].fill_(beta)
        # nudge for classification
        if self.trainclassifier:
            self.betas[-1].fill_(beta)

        return self.betas

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
 
        # not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     

        
        self.targetneurons = self.xytotargetneurons(x, y)
        self.beta = self.fieldbeta(beta)

        return self.energymodel(self.targetneurons, neurons, T, self.betas, self.fullclamping, criterion, check_thm=check_thm)

    def energymodel(self, targetneurons, neurons, T, betas, fullclamping, criterion, check_thm=False):
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = neurons[0].size(0)
        device = neurons[0].device
        neuronsout = []
        for idx in range(len(neurons)):
            neuronsout.append(torch.zeros_like(neurons[idx]))
            neuronsout[idx].requires_grad = True
        # apply full clamping
        with torch.no_grad():
            for idx in range(len(neurons)):
                if fullclamping[idx].size(0) >  0:
                    neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]
                neurons[idx].requires_grad = True
        for t in range(T):
            phi = self.Phi(targetneurons, neurons, betas, fullclamping, criterion)
            grads = torch.autograd.grad(phi, neurons, grad_outputs=self.initgradstensor, create_graph=check_thm)

            for idx in range(0,len(neurons)-1):
                neurons[idx] = self.activation( grads[idx] )
                # if check_thm:
                #   neurons[idx].retain_grad()
                # else:
                neurons[idx].requires_grad = True
         
            if not_mse and not(self.softmax):
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation( grads[-1] )
            # if chek_thm:
            #   neurons[-1].retain_grad()
            neurons[-1].requires_grad = True

            # apply full clamping
            with torch.no_grad():
                for idx in range(len(neurons)):
                    if fullclamping[idx].size(0) >  0:
                        neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]

            ## average the last n neurons in case its a limit cycle
            #n = 3
            #if T-t <= n:
            #    for idx in range(len(neurons)):
            #        neuronsout[idx] += neurons[idx]/n

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
        for idx in range(len(neurons)):
            self.targetneurons.append(torch.tensor([], device=device))
            self.betas.append(torch.tensor([], device=device))
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



class RevLatCNN(Reversible_CNN):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, lat_layer_idxs, activation=my_sigmoid, softmax=False):
        super(RevLatCNN, self).__init__(in_size, channels, kernels, strides, fc, pools, paddings, activation=activation, softmax=False)

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



    def Phi(self, targetneurons, neurons, betas, fullclamping, criterion):
        phi = Reversible_CNN.Phi(self, targetneurons, neurons, betas, fullclamping, criterion)
        mbs = neurons[0].size(0)
        #conv_len = len(self.kernels)
        #tot_len = len(self.synapses)

        ##layers = [x] + neurons        
        ##layers = neurons
        #phi = 0.0

        #for idx in range(conv_len):    
        #    phi += torch.sum( self.pools[idx](self.synapses[idx](neurons[idx])) * neurons[idx+1], dim=(1,2,3)).squeeze()     
        for j, idx in enumerate(self.lat_layer_idxs):
            phi += torch.sum( self.lat_syn[j](neurons[idx].view(mbs,-1)) * neurons[idx].view(mbs,-1), dim=1).squeeze()
        ##Phi computation changes depending on softmax == True or not
        #if not self.softmax:
        #    for idx in range(conv_len, tot_len):
        #        phi += torch.sum( self.synapses[idx](neurons[idx].view(mbs,-1)) * neurons[idx+1], dim=1).squeeze()
        #else:
        #    # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
        #    for idx in range(conv_len, tot_len-1):
        #        phi += torch.sum( self.synapses[idx](neurons[idx].view(mbs,-1)) * neurons[idx+1], dim=1).squeeze()
        #     
        #for idx in range(len(neurons)):
        #    if targetneurons[idx].size(0) > 0:
        #        # if fullclamping[idx].size(0) >  0:
        #        #     neurons[idx][fullclamping[idx]] = targetneurons[idx][fullclamping[idx]]
        #        phi -= betas[idx]*criterion(neurons[idx], targetneurons[idx]).sum().squeeze()             

        return phi
