import glob
import torch
# import re
import argparse

# copied from main.py
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
parser.add_argument('--lat-constraints', nargs='+', type = str, default = [], metavar = 'lc', help='constraints to impose to lateral connections (layer-wise). e.g. `--lat-constraints zerodiag transposesymmetric+negReLu none` to zero the diagonal (self-interactions) of the first lateral layer in the model, and ensure the second layer is a symmetric matrix with only negative values (this combines two possible constraints).')
parser.add_argument('--competitiontype', type = str, default = 'none', metavar = 'ct', help='(LatSoftCNN) type of lateral inhibition to apply in output classification layer (feature_inner_products or uniform_inhbition). will be scaled by --inhibitstrength')
parser.add_argument('--inhibitstrength',type = float, default = 0.0, metavar = 'inhibitstrength', help='(LatSoftCNN) coeffecient on WTA inhibitory connection in output layer (mimicing softmax)')
parser.add_argument('--lat-init-zeros', default = False, action = 'store_true', help='whether to initialze the lateral/hopfield interactions with zeros (default: False)')
parser.add_argument('--lat-lrs', nargs='+', type = float, default = [], metavar = 'l', help='lateral connection set wise lr')
parser.add_argument('--head-lrs', nargs='+', type = float, default = [], metavar = 'hl', help='(multi-head CNN) head-encoder-layer wise lr')
parser.add_argument('--lat-wds', nargs='+', type = float, default = None, metavar = 'l', help='lateral connection set weight decays')

parser.add_argument('--save-nrn', default = False, action = 'store_true', help='not sure what this is supposed to be for. it was in the check/*.sh, but not in main.py so it originally errored.')

parser.add_argument('--load-path-convert', type = str, default = '', metavar = 'l', help='load a model and copy its parameters to the specified architecture (initialize new layers at identity)')
parser.add_argument('--convert-place-layers', nargs='+', type = int, default = [], help='index of layers to convert from loaded model (indexes not specified should be linear layers and architecture should match original at given indexes)')

parser.add_argument('--tensorboard', default = False, action = 'store_true', help='write data to tensorboard for viewing while training')


def getruns(pathpattern = 'results/*/*/*/*/'):
    runs = []
    for rundir in glob.glob(pathpattern):
        for checkptfile in glob.glob(rundir+'checkpoint.tar'):
            statedict = torch.load(checkptfile)
            with open(rundir + 'hyperparameters.txt', 'r') as paramfile:
                callline = paramfile.readline()
#             runs.append({'train':      statedict['train_acc'][-1],
#                          'test':       statedict['test_acc'][-1],
#                          'epochs':     statedict['epoch'],
#                          'task':       ', '.join(re.findall('--task (\w+)', callline)),
#                          'model':      ', '.join(re.findall('--model (\w+)', callline)),
#                          'lateral':    ', '.join(re.findall('(--use-lat|--train-lat)', callline)),
#                          'softmax':    ', '.join(re.findall('(--softmax)', callline)),
#                          'thirdphase': ', '.join(re.findall('(--thirdphase)', callline)),
#                          'randomsign': ', '.join(re.findall('(--random-sign)', callline)),
#                          'loss':       ', '.join(re.findall('--loss (\w+)', callline)),
#                          'activation': ', '.join(re.findall('--act (\w+)', callline)),
#                          'beta1':      ', '.join(re.findall('--betas ([^ ]+) (?:[^ ]+)', callline)),
#                          'beta2':      ', '.join(re.findall('--betas (?:[^ ]+) ([^ ]+)', callline)),
#                          'lat-init':   ', '.join(re.findall('--lat-init-(\w+)', callline)),
#                          'load':       ', '.join(re.findall('--load-path-convert ([^ ]*)', callline)),
#                          'dir':        rundir, 'callline': callline})
            runinfo = vars(parser.parse_args(callline.strip().split(' ')[1:]))
            runinfo['train'] = statedict['train_acc'][-1]
            runinfo['test'] = statedict['test_acc'][-1]
            runinfo['dir'] = rundir
            runinfo['call'] = callline
            runs.append(runinfo)
    return runs