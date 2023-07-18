import glob
import torch
import re


def getruns(pathpattern = 'results/*/*/*/*/'):
    runs = []
    for rundir in glob.glob(pathpattern):
        for checkptfile in glob.glob(rundir+'checkpoint.tar'):
            statedict = torch.load(checkptfile)
            with open(rundir + 'hyperparameters.txt', 'r') as paramfile:
                callline = paramfile.readline()
            runs.append({'train':      statedict['train_acc'][-1],
                         'test':       statedict['test_acc'][-1],
                         'epochs':     statedict['epoch'],
                         'task':       ', '.join(re.findall('--task (\w+)', callline)),
                         'model':      ', '.join(re.findall('--model (\w+)', callline)),
                         'lateral':    ', '.join(re.findall('(--use-lat)', callline)),
                         'softmax':    ', '.join(re.findall('(--softmax)', callline)),
                         'thirdphase': ', '.join(re.findall('(--thirdphase)', callline)),
                         'randomsign': ', '.join(re.findall('(--random-sign)', callline)),
                         'loss':       ', '.join(re.findall('--loss (\w+)', callline)),
                         'activation': ', '.join(re.findall('--act (\w+)', callline)),
                         'beta1':      ', '.join(re.findall('--betas ([^ ]+) (?:[^ ]+)', callline)),
                         'beta2':      ', '.join(re.findall('--betas (?:[^ ]+) ([^ ]+)', callline)),
                         'lat-init':   ', '.join(re.findall('--lat-init-(\w+)', callline)),
                         'load':       ', '.join(re.findall('--load-path-convert ([^ ]*)', callline)),
                         'dir':        rundir, 'callline': callline})
    return runs