import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import argparse
import matplotlib
matplotlib.use('Agg')

import glob
from PIL import Image
import os
from datetime import datetime
import time
import math
import sys
from model_utils import *
from data_utils import *

from lateral_models import *

from hopfield_models import *

parser = argparse.ArgumentParser(description='Eqprop')
parser.add_argument('--model',type = str, default = 'MLP', metavar = 'm', help='model')
parser.add_argument('--task',type = str, default = 'MNIST', metavar = 't', help='task')

parser.add_argument('--pools', type = str, default = 'mm', metavar = 'p', help='pooling')
parser.add_argument('--archi', nargs='+', type = int, default = [784, 512, 10], metavar = 'A', help='architecture of the network')
parser.add_argument('--channels', nargs='+', type = int, default = [32, 64], metavar = 'C', help='channels of the convnet')
parser.add_argument('--kernels', nargs='+', type = int, default = [5, 5], metavar = 'K', help='kernels sizes of the convnet')
parser.add_argument('--strides', nargs='+', type = int, default = [1, 1], metavar = 'S', help='strides of the convnet')
parser.add_argument('--paddings', nargs='+', type = int, default = [0, 0], metavar = 'P', help='paddings of the conv layers')
parser.add_argument('--fc', nargs='+', type = int, default = [10], metavar = 'S', help='linear classifier of the convnet')

parser.add_argument('--act',type = str, default = 'mysig', metavar = 'a', help='activation function')
parser.add_argument('--optim', type = str, default = 'sgd', metavar = 'opt', help='optimizer for training')
parser.add_argument('--lrs', nargs='+', type = float, default = [], metavar = 'l', help='layer wise lr')
parser.add_argument('--wds', nargs='+', type = float, default = None, metavar = 'l', help='layer weight decays')
parser.add_argument('--mmt',type = float, default = 0.0, metavar = 'mmt', help='Momentum for sgd')
parser.add_argument('--loss', type = str, default = 'mse', metavar = 'lss', help='loss for training')
parser.add_argument('--alg', type = str, default = 'EP', metavar = 'al', help='EP or BPTT or CEP')
parser.add_argument('--mbs',type = int, default = 20, metavar = 'M', help='minibatch size')
parser.add_argument('--T1',type = int, default = 20, metavar = 'T1', help='Time of first phase')
parser.add_argument('--T2',type = int, default = 4, metavar = 'T2', help='Time of second phase')
parser.add_argument('--betas', nargs='+', type = float, default = [0.0, 0.01], metavar = 'Bs', help='Betas')
parser.add_argument('--epochs',type = int, default = 1,metavar = 'EPT',help='Number of epochs per tasks')
parser.add_argument('--check-thm', default = False, action = 'store_true', help='checking the gdu while training')
parser.add_argument('--random-sign', default = False, action = 'store_true', help='randomly switch beta_2 sign')
parser.add_argument('--data-aug', default = False, action = 'store_true', help='enabling data augmentation for cifar10')
parser.add_argument('--lr-decay', default = False, action = 'store_true', help='enabling learning rate decay')
parser.add_argument('--scale',type = float, default = None, metavar = 'g', help='scal factor for weight init')
parser.add_argument('--save', default = False, action = 'store_true', help='saving results')
parser.add_argument('--todo', type = str, default = 'train', metavar = 'tr', help='training or plot gdu curves')
parser.add_argument('--load-path', type = str, default = '', metavar = 'l', help='load a model')
parser.add_argument('--seed',type = int, default = None, metavar = 's', help='random seed')
parser.add_argument('--device',type = int, default = 0, metavar = 'd', help='device')
parser.add_argument('--thirdphase', default = False, action = 'store_true', help='add third phase for higher order evaluation of the gradient (default: False)')
parser.add_argument('--softmax', default = False, action = 'store_true', help='softmax loss with parameters (default: False)')
parser.add_argument('--same-update', default = False, action = 'store_true', help='same update is applied for VFCNN back and forward')
parser.add_argument('--cep-debug', default = False, action = 'store_true', help='debug cep')


parser.add_argument('--train-lateral', default = False, action = 'store_true', help='whether to enable the lateral/hopfield interactions (default: False)')
parser.add_argument('--lat-layers', nargs='+', type = int, default = [], help='index of layers to add lateral connections to (ex: 0 1 -1) to add lateral interactions to first and second layer, last layer')
parser.add_argument('--lat-kernels', nargs='+', type = int, default = [], help='kernel size of convolutional lateral interaction. Must be odd numbers. Should have length equal to the number of --lat-layers indexes which have a convolutional input. Other layer indexes will be fully connected (like their input). For use with LatConvCNN')
parser.add_argument('--sparse-layers', nargs='+', type = int, default = [], help='index of layers to add lateral connections (--competition-type) to, and a fixed bias of -(--lambdas)')
parser.add_argument('--lat-constraints', nargs='+', type = str, default = [], metavar = 'lc', help='constraints to impose to lateral connections (layer-wise). e.g. `--lat-constraints zerodiag transposesymmetric+negReLu none` to zero the diagonal (self-interactions) of the first lateral layer in the model, and ensure the second layer is a symmetric matrix with only negative values (this combines two possible constraints).')
parser.add_argument('--comp-syn-constraints', nargs='+', type = str, default = [], metavar = 'lc', help='constraints to impose on competitive sparsifying lateral connections (layer-wise). e.g. `--lat-constraints zerodiag transposesymmetric+negReLu none` to zero the diagonal (self-interactions) of the first lateral layer in the model, and ensure the second layer is a symmetric matrix with only negative values (this combines two possible constraints).')
parser.add_argument('--competitiontype', type = str, default = 'none', metavar = 'ct', help='(LatSoftCNN) type of lateral inhibition to apply in output classification layer (feature_inner_products or uniform_inhbition). will be scaled by --inhibitstrength')
parser.add_argument('--inhibitstrength',type = float, default = 0.0, metavar = 'inhibitstrength', help='(LatSoftCNN) coeffecient on WTA inhibitory connection in output layer (mimicing softmax)')
parser.add_argument('--lat-init-zeros', default = False, action = 'store_true', help='whether to initialze the lateral/hopfield interactions with zeros (default: False)')
parser.add_argument('--lat-lrs', nargs='+', type = float, default = [], metavar = 'l', help='lateral connection set wise lr')
parser.add_argument('--head-lrs', nargs='+', type = float, default = [], metavar = 'hl', help='(multi-head CNN) head-encoder-layer wise lr')
parser.add_argument('--lat-wds', nargs='+', type = float, default = None, metavar = 'l', help='lateral connection set weight decays')

parser.add_argument('--save-nrn', default = False, action = 'store_true', help='not sure what this is supposed to be for. it was in the check/*.sh, but not in main.py so it originally errored.')

parser.add_argument('--load-path-convert', type = str, default = '', metavar = 'l', help='load a model and copy its parameters to the specified architecture (initialize new layers at identity)')
parser.add_argument('--convert-place-layers', nargs='+', type = str, default = [], help='index of layers to convert from loaded model. use `-` as i-th input if i-th layer of original model shouldnt be used. (indexes not specified should be linear layers and architecture should match original at given indexes)')

parser.add_argument('--tensorboard', default = False, action = 'store_true', help='write data to tensorboard for viewing while training')

parser.add_argument('--lambdas', nargs='+', type = float, default=[], help='(sparse-)layer-wise fixed value for negative bias. Equivelant to L1 penalty on neurons scaled by this.')
parser.add_argument('--nudge-lambda', nargs='+', type = float, default=[], help='impose L1 penalty in nudged phase with this coefficient, to train for sparsity with EP')

parser.add_argument('--eps', nargs='+', type = float, default = [], metavar = 'e', help='epsilon values to use for PGD attack (--todo attack)')
parser.add_argument('--mbs-test',type = int, default = 200, metavar = 'M', help='minibatch size for test set (can be larger since during testing grads need not be calculated)')
parser.add_argument('--nbatches',type = int, default = 20, metavar = 'M', help='maximum number of batches to make adversarial examples of')
parser.add_argument('--figs', default = False, action='store_true', help='plot and save figures')

parser.add_argument('--jit', default = False, action = 'store_true', help='use torch.jit trace and script to try to optimize the code for CUDA')
parser.add_argument('--cpu', default = False, action = 'store_true', help='use CPU rather than CUDA')

parser.add_argument('--noise', type=float, default = 0.0,  help='standard deviation of guassian noise added to training data')

parser.add_argument('--dt', type=float, default = 1.0,  help='timestep for model dynamics. Decay time constant = 1.0')

args = parser.parse_args()
command_line = ' '.join(sys.argv)

print('\n')
print(command_line)
print('\n')
print('##################################################################')
print('\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas')
print('\t',args.mbs,'\t',args.T1,'\t',args.T2,'\t',args.epochs,'\t',args.act, '\t', args.betas)
print('\n')

device = torch.device('cuda:'+str(args.device) if (not args.cpu) and torch.cuda.is_available() else 'cpu')


if args.save:
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H-%M-%S')
    if args.load_path=='':
        path = 'results/'+args.alg+'/'+args.loss+'/'+date+'/'+time+'_gpu'+str(args.device)
    else:
        path = args.load_path
    if not(os.path.exists(path)):
        os.makedirs(path)
    print('---------- saving at {} --------------'.format(path))
else:
    path = ''


#if args.alg=='CEP' and args.cep_debug:
#    torch.set_default_dtype(torch.float64)

print('Default dtype :\t', torch.get_default_dtype(), '\n')


mbs=args.mbs
if args.seed is not None:
    torch.manual_seed(args.seed)




if args.task=='MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=mbs, shuffle=True, num_workers=0)

    mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=args.mbs_test, shuffle=False, num_workers=0)

elif args.task=='CIFAR10':
    if args.data_aug:
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])
    else:
         transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])   
    if args.noise > 0.0:
        transform_train = torchvision.transforms.Compose([transform_train, AddGaussianNoise(0.0, args.noise)])

    if args.todo=='attack':
        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    else:
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
    test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=args.mbs_test, shuffle=False, num_workers=1)




if args.act=='mysig':
    activation = my_sigmoid
elif args.act=='sigmoid':
    activation = torch.sigmoid
elif args.act=='tanh':
    activation = torch.tanh
elif args.act=='hard_sigmoid':
    activation = hard_sigmoid
elif args.act=='my_hard_sig':
    activation = my_hard_sig
elif args.act=='ctrd_hard_sig':
    activation = ctrd_hard_sig




if args.loss=='mse':
    criterion = torch.nn.MSELoss(reduction='none').to(device)
elif args.loss=='cel':
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
elif args.loss=='loggrad':
    # equivelant to CEL if output was already softmaxed.
    # instead of taking softmax of unconstrinaed neurons, the denominator of the softmax is just a term in the loss.
    # if logsumexp doesn't work for some reason (unstable), could use L1 penalty for incorrect neurons
    criterion = lambda output, target: -(output[target]).sum(dim=1) + torch.logsumexp(output, dim=1) #+ (output.sum(dim=1) - output[target].sum(dim=1))
print('loss =', criterion, '\n')



if args.load_path=='':

    if args.model=='MLP':
        model = P_MLP(args.archi, activation=activation)
    elif args.model=='AnalyticalMLP':
        model = MLP_Analytical(args.archi, activation=activation)
    elif args.model=='VFMLP':
        model = VF_MLP(args.archi, activation=activation)
    elif args.model=='LatMLP':
        model = Lat_P_MLP(args.archi, activation=activation)
    elif args.model.find('CNN')!=-1:

        pools = make_pools(args.pools)
        if args.task=='MNIST':
            channels = [1]+args.channels 
            in_size = 28
        elif args.task=='CIFAR10':
            channels = [3]+args.channels
            in_size = 32
        elif args.task=='imagenet':   #only for gducheck
            pools = make_pools(args.pools)
            channels = [3]+args.channels 
            model = P_CNN(224, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                            activation=activation, softmax=args.softmax)
            
        if args.model=='CNN':
            model = P_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                              activation=activation, softmax=args.softmax)
        elif args.model=='VFCNN':
            model = VF_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                               activation=activation, softmax=args.softmax, same_update=args.same_update)
        elif args.model=='Lat_MH_CNN':
            model = Lat_MH_CNN([50 for i in range(10)], in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                                activation=activation, softmax=args.softmax, same_update=args.same_update)
        elif args.model=='LatSoftCNN':
            model = fake_softmax_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                              activation=activation, competitiontype=args.competitiontype, lat_constraints=args.lat_constraints, 
                              inhibitstrength=args.inhibitstrength, softmax=False)
        elif args.model=='SparseCodingCNN':
            model = latCompCNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                              lat_layer_idxs=args.lat_layers, sparse_layer_idxs=args.sparse_layers, comp_syn_constraints = args.comp_syn_constraints,
                              competitiontype=args.competitiontype, lat_constraints=args.lat_constraints,
                              inhibitstrength=args.inhibitstrength, activation=activation, softmax=args.softmax, layerlambdas=args.lambdas, dt=args.dt)
        elif args.model=='ReversibleCNN':
            model = Reversible_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                              activation=activation, softmax=args.softmax)
        elif args.model=='DenoiseLGNCNN':
            model = Reversible_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                              activation=activation, softmax=args.softmax, dt=args.dt)
        elif args.model=="LateralCNN":
            model = lat_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                            args.lat_layers, lat_constraints=args.lat_constraints,
                          activation=activation, softmax=args.softmax)
        elif args.model=="LatConvCNN":
            model = lat_conv_CNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                            args.lat_kernels, args.lat_layers, lat_constraints=args.lat_constraints,
                          activation=activation, softmax=args.softmax)
        elif args.model=="RevLatCNN":
            model = RevLatCNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                            args.lat_layers, lat_constraints = args.lat_constraints,
                          activation=activation, softmax=args.softmax)
        elif args.model=="HopfieldCNN":
            model = HopfieldCNN(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                            args.lat_layers, lat_constraints = args.lat_constraints,
                          activation=activation, softmax=args.softmax, criterion=criterion)
        if args.model=='StochasticCNN':
            model = P_CNN_Stochastic(in_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                              activation=activation, softmax=args.softmax, dynamicsmomentum=0.5)

                       
        if args.load_path_convert != '':
            # initialize new weights as identity
            def my_convert_init(m): # initialize new layers as identity, so model functions like original model
                if isinstance(m, torch.nn.Linear):
                    m.weight.data.copy_(torch.eye(m.weight.size(0), m.weight.size(1)))
                    if m.bias is not None:
                        m.bias.data.zero_()
            model.synapses.apply(my_convert_init)
            
            # load source model
            origmodel = torch.load(args.load_path_convert + '/model.pt', map_location=device)
            # copy parameters
            if len(args.convert_place_layers) == 0:
                args.convert_place_layers = range(len(origmodel.synapses))
            for i, idx in enumerate(args.convert_place_layers):
                if idx != '-':
                    idx = int(idx)
                    model.synapses[idx] = origmodel.synapses[i]
            #if origmodel.hasattr('lat_syn'):
            #    if len(args.convert_place_lat_layers) == 0:
            #        args.convert_place_lat_layers = range(len(origmodel.synapses))
            #    model.lat_syn[args.convert_place_lat_layers] = origmodel.lat_syn

            # modify lateral or tranposed layers if necessary based on sourced weights
            if hasattr(model, 'postupdate'):
                model.postupdate()
        else:
            if not(args.train_lateral) or args.lat_init_zeros:
                if hasattr(model, 'lat_syn'):
                    print("zeroing lateral layers")
                    for l in model.lat_syn:
                        l.weight.data = l.weight.data.zero_()
                        if hasattr(l, 'bias') and l.bias is not None:
                            l.bias.data = l.bias.data.zero_()
                if hasattr(model, 'conv_lat_syn'):
                    print("zeroing conv lateral layers")
                    for l in model.conv_lat_syn:
                        l.weight.data = l.weight.data.zero_()
                        if hasattr(l, 'bias') and l.bias is not None:
                            l.bias.data = l.bias.data.zero_()
                elif args.model == 'Lat_MH_CNN':
                    print("zeroing head lateral layers")
                    model.head_hopfield.apply(torch.nn.init.zeros_)

        print('\n')
        print('Poolings =', model.pools)

    if args.scale is not None:
        model.apply(my_init(args.scale))
else:
    model = torch.load(args.load_path + '/model.pt', map_location=device)

model.to(device)
print(model)

betas = args.betas[0], args.betas[1]


if args.todo=='train':
    assert(len(args.lrs)==len(model.synapses))

    # Constructing the optimizer
    optim_params = []
    if (args.alg=='CEP' and args.wds) and not(args.cep_debug):
        for idx in range(len(model.synapses)):
            args.wds[idx] = (1 - (1 - args.wds[idx] * 0.1 )**(1/args.T2))/args.lrs[idx]

    for idx in range(len(model.synapses)):
        if args.wds is None:
            optim_params.append(  {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx]}  )
        else:
            optim_params.append(  {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx], 'weight_decay': args.wds[idx]}  )
    if hasattr(model, 'B_syn'):
        for idx in range(len(model.B_syn)):
            if args.wds is None:
                optim_params.append( {'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx+1]} )
            else:
                optim_params.append( {'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx+1], 'weight_decay': args.wds[idx+1]} )
    if hasattr(model, 'head_encoders'):
        for idx in range(len(model.head_encoders)):
            if args.wds is None:
                optim_params.append( {'params': model.head_encoders[idx].parameters(), 'lr': args.head_lrs[idx]} )
            else:
                optim_params.append( {'params': model.head_encoders[idx].parameters(), 'lr': args.head_lrs[idx], 'weight_decay': args.head_wds[idx+1]} )
    print('trian lat?', args.train_lateral, hasattr(model, 'lat_syn'))
    if args.train_lateral:
        if hasattr(model, 'conv_lat_syn'):
            print('adding lateral synapses to optimizer')
            for idx in range(len(model.conv_lat_syn)):
                if args.wds is None:
                    optim_params.append( {'params': model.conv_lat_syn[idx].parameters(), 'lr': args.lat_lrs[idx]} )
                else:
                    optim_params.append( {'params': model.conv_lat_syn[idx].parameters(), 'lr': args.lat_lrs[idx], 'weight_decay': args.lat_wds[idx]} )
            conv_lat_len = idx
            for j in range(len(model.lat_syn)):
                idx = conv_lat_len+j
                if args.wds is None:
                    optim_params.append( {'params': model.lat_syn[j].parameters(), 'lr': args.lat_lrs[idx]} )
                else:
                    optim_params.append( {'params': model.lat_syn[j].parameters(), 'lr': args.lat_lrs[idx], 'weight_decay': args.lat_wds[idx]} )
        elif hasattr(model, 'lat_syn'):
            print('adding lateral synapses to optimizer')
            for idx in range(len(model.lat_syn)):
                if args.wds is None:
                    optim_params.append( {'params': model.lat_syn[idx].parameters(), 'lr': args.lat_lrs[idx]} )
                else:
                    optim_params.append( {'params': model.lat_syn[idx].parameters(), 'lr': args.lat_lrs[idx], 'weight_decay': args.lat_wds[idx]} )
        if hasattr(model, 'head_hopfield'):
            for idx in range(len(model.head_hopfield)):
                if args.wds is None:
                    optim_params.append( {'params': model.head_hopfield[idx].parameters(), 'lr': args.lat_lrs[idx]} )
                else:
                    optim_params.append( {'params': model.head_hopfield[idx].parameters(), 'lr': args.lat_lrs[idx], 'weight_decay': args.lat_wds[idx]} )

    if args.optim=='sgd':
        optimizer = torch.optim.SGD( optim_params, momentum=args.mmt )
    elif args.optim=='adam':
        optimizer = torch.optim.Adam( optim_params )

    if model.__class__.__name__ == 'sparseCNN' and args.train_lateral:
        model.lambdas = args.lambdas
        
        sparse_op = []
        idx = 0
        for modulelist in (model.conv_comp_layers, model.fc_comp_layers):
            for layer in modulelist:
                if args.lat_wds is None:
                    sparse_op.append( {'params': layer.parameters(), 'lr': args.lat_lrs[idx]} )
                else:
                    sparse_op.append( {'params': layer.parameters(), 'lr': args.lat_lrs[idx], 'weight_decay': args.lat_wds[idx]} )
            idx += 1

        model.sparse_optim = torch.optim.SGD( sparse_op, momentum=args.mmt )
        if args.lr_decay:
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)
            model.sparse_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.sparse_optim, 100, eta_min=1e-5)


    # Constructing the scheduler
    if args.lr_decay:
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)
    else:
        scheduler = None

    # Loading the state when resuming a run
    if args.load_path!='':
        checkpoint = torch.load(args.load_path + '/checkpoint.tar')
        optimizer.load_state_dict(checkpoint['opt'])
        if 'sparse_optim' in checkpoint.keys():
            opt_sparse.load_state_dict(checkpoint['sparse_optim'])
        if checkpoint['scheduler'] is not None and args.lr_decay:
            scheduler.load_state_dict(checkpoint['scheduler'])
    elif args.load_path_convert!='':
        checkpoint = torch.load(args.load_path_convert + '/checkpoint.tar')
        optimizer.load_state_dict(checkpoint['opt'])
        if 'sparse_optim' in checkpoint.keys():
            opt_sparse.load_state_dict(checkpoint['sparse_optim'])
        if checkpoint['scheduler'] is not None and args.lr_decay:
            scheduler.load_state_dict(checkpoint['scheduler'])
    else: 
        checkpoint = None
    
    print(optimizer)
    print('\ntraining algorithm : ',args.alg, '\n')
    if args.save and args.load_path=='':
        createHyperparametersFile(path, args, model, command_line)
        
    if args.jit:
        print('Tracing forwards with JIT')
        x = torch.rand((mbs, channels[0], in_size, in_size), device=device)
        y = torch.round(9*torch.rand((mbs,))).to(device)
        # yhat = torch.round(9*torch.rand((mbs,))).to(device)
        neurons = model.init_neurons(mbs, device)
        [n.requires_grad_(False) for n in neurons]
        print('n0 grad?', neurons[0].requires_grad)
        T = torch.tensor(1).to(device)
        T = 10
        beta = torch.tensor(0.1).to(device)
        model = model.to(device)
        # jitcrit = torch.jit.trace(criterion, (yhat, y))
        # invaliddata = torch.zeros((1,)).to(device) # this is passed to the unused criterion argument placed for compatability
        print('original forward', model.forward)
        #with torch.jit.optimized_execution(True):
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True, record_shapes=True) as prof:
        # torch.jit.optimize_for_inference
        jitforward = (torch.jit.script(model))#, example_inputs={
                                #model.forward: (model, x, y, neurons, T, beta) })).eval().forward
        #def fuckingstopgraddingmyshit(*args, **kwargs):
        #    with torch.no_grad():
        #        neurons=jitforward(*args, **kwargs)
        #    return neurons
        #model.forward = fuckingstopgraddingmyshit
                # {'x':x, 'y':y, 'neurons':neurons, 'T':T, 'beta':beta}
        print('jit-traced forward', model.forward)
        args.T1 = torch.as_tensor(args.T1).to(device)
        args.T2 = torch.as_tensor(args.T1).to(device)
        betas = torch.as_tensor(betas).to(device)


    train(model, optimizer, train_loader, test_loader, args.T1, args.T2, betas, device, args.epochs, criterion, alg=args.alg, 
                 random_sign=args.random_sign, check_thm=args.check_thm, save=args.save, path=path, checkpoint=checkpoint, 
                 thirdphase=args.thirdphase, scheduler=scheduler, cep_debug=args.cep_debug, tensorboard=args.tensorboard)


elif args.todo=='attack':
    print('Performing PGD attacks on model')
    savepath = path+'/PGD_attack/'
    os.makedirs(savepath, exist_ok=True)
    accs = []
    accs_adv = []
    for eps in args.eps:
        print('Now attacking with epsilon : ', eps)
        acc, acc_adv, examples, preds, preds_adv = attack(model, test_loader, args.nbatches, args.T1, args.T2, eps, criterion, device, savepath, save_adv_examples=args.save, figs=args.figs)     
        accs.append(acc)
        accs_adv.append(acc_adv)
    if args.save:
        np.save(savepath + '/attacked_accuracy.npy', np.asarray([args.eps, accs_adv]))
    if args.figs and len(accs) > 1:
        fig = plt.figure()
        plt.plot(args.eps, accs, label='original accuracy', linestyle='--')
        plt.plot(args.eps, accs_adv, label='attacked')
        plt.title('accuracy vs. strength of PGD attack')
        plt.xlabel('epsilon')
        plt.ylabel('accuracy')
        plt.legend()
        fig.savefig(savepath + '/robustness.pdf', bbox_inches='tight')
elif args.todo=='gducheck':
    RMSE(BPTT, EP)
    if args.save:
        bptt_est = get_estimate(BPTT) 
        ep_est = get_estimate(EP)
        torch.save(bptt_est, path+'/bptt.tar')
        torch.save(BPTT, path+'/BPTT.tar')
        torch.save(ep_est, path+'/ep.tar') 
        torch.save(EP, path+'/EP.tar') 
        if args.thirdphase:
            ep_2_est = get_estimate(EP_2)
            torch.save(ep_2_est, path+'/ep_2.tar')
            torch.save(EP_2, path+'/EP_2.tar')
            compare_estimate(bptt_est, ep_est, ep_2_est, path)
            plot_gdu(BPTT, EP, path, EP_2=EP_2, alg=args.alg)
        else:
            plot_gdu(BPTT, EP, path, alg=args.alg)


elif args.todo=='evaluate':

    training_acc = evaluate(model, train_loader, args.T1, device)
    training_acc /= len(train_loader.dataset)
    print('\nTrain accuracy :', round(training_acc,2), file=open(path+'/hyperparameters.txt', 'a'))
    test_acc = evaluate(model, test_loader, args.T1, device)
    test_acc /= len(test_loader.dataset)
    print('\nTest accuracy :', round(test_acc, 2), file=open(path+'/hyperparameters.txt', 'a'))








