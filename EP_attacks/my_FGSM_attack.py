import torch

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def attack(model, args, epsilon=0.01, attacks=10):
    mbs = args.mbs.item()
    beta = 0.0
    model = model.to(device)
    
    if args.loss.item()=='mse':
        criterion = torch.nn.MSELoss(reduction='none').to(device)
    elif args.loss.item()=='cel':
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    
    Tatt = 10
    Tleft = 250 - Tatt
    
    attackxs = []
    origxs = []
    origpreds = []
    attackedpreds = []
    truthlabels = []
    for s in range(attacks):
        x, y = next(iter(train_loader))
        x = x.to(device)
        x = x.requires_grad_()

        y = y.to(device)
    
    
        neurons = model.init_neurons(mbs, device)
        neurons = model(x, y, neurons, Tatt, beta, criterion, check_thm=True) # check_thm True so grads are retaiend
        xnograd = x.requires_grad_(False)
        neurons = model(xnograd, y, neurons, Tleft, beta, criterion, check_thm=True) # check_thm True so grads are retaiend
        
        if args.softmax.item():
            # the prediction is made with softmax[last weights[penultimate layer]]
            pred = model.synapses[-1](neurons[-1].view(mbs,-1))
        else:
            pred = neurons[-1]
        
        origcorrect = torch.eq(pred.max(1).indices, y)
        print('original acc : ', (torch.sum(origcorrect)/mbs).item(), end='\t')

        
        if criterion.__class__.__name__.find('MSE')!=-1:
            L = 0.5*criterion(pred.float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).squeeze()
        else:
            L = criterion(pred.float(), y).squeeze()
        
        model.zero_grad()
        L.sum().backward()
        attackx = fgsm_attack(x, epsilon, x.grad).detach()
        
        
        neurons = model.init_neurons(mbs, device)
        neurons = model(attackx, y, neurons, Tatt+Tleft, beta, criterion, check_thm=True) # check_thm True so grads are retaiend
#         xnograd = x.requires_grad_(False)
#         neurons = model(xnograd, y, neurons, Tleft, beta, criterion, check_thm=True) # check_thm True so grads are retaiend
        
        if args.softmax.item():
            # the prediction is made with softmax[last weights[penultimate layer]]
            attackpred = model.synapses[-1](neurons[-1].view(mbs,-1))
        else:
            attackpred = neurons[-1]
        
        attackedcorrect = torch.eq(attackpred.max(1).indices, y)
        print('attacked acc : ', (torch.sum(attackedcorrect)/mbs).item())
        
        attacksuccess = torch.logical_and(False == attackedcorrect, True == origcorrect)
        attackxs.append(attackx[attacksuccess].detach())
        origxs.append(x[attacksuccess].detach())
        origpreds.append(pred[attacksuccess].detach())
        attackedpreds.append(attackpred[attacksuccess].detach())
        truthlabels.append(y[attacksuccess].detach())
        
        del neurons, x, xnograd, y, pred, attachpred
            
    return attackxs, origxs, attackedpreds, origpreds, truthlabels