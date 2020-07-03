
import torch
import pdb
import torch.nn.functional as F

def mse_loss(input, target, mask=None, needSigmoid=True):
    if needSigmoid:
        input = torch.sigmoid(input)
    if mask is not None:
        
        input = input * mask
        #target = target * mask
    loss = F.mse_loss(input, target)
    return loss

def nmse_loss(input, target, mask=None, needSigmoid=True):
    '''
    train the 2D CNN based model
    '''
    if needSigmoid:
        input = torch.sigmoid(input)
    if mask is not None:
        input = input * mask
    res = input-target
    res_norm = torch.norm(res, dim=(2,3))
    res_norm = res_norm**2
    res_norm = torch.sum(res_norm, dim=1)
    #res_norm = torch.sqrt(res_norm)
    target_norm = torch.norm(target, dim=(2,3))
    target_norm = target_norm**2
    target_norm = torch.sum(target_norm, dim=1)
    #target_norm = torch.sqrt(target_norm)
    nmse = res_norm/target_norm
    return torch.mean(nmse)

def nmse_loss_v2(input, target):
    '''
    train the LISTA model, batch size is at axis 1
    '''
    res = input - target
    res_norm = torch.norm(res, dim=0)**2
    target_norm = torch.norm(target, dim=0)**2
    mask = target_norm != 0
    nmse = res_norm[mask] /target_norm[mask]
    return torch.mean(nmse)
    

def bce_loss(input, target, needSigmoid=True):
    pos_weight = torch.Tensor([0.05]).to(input.device)
    if needSigmoid:
        return F.binary_cross_entropy_with_logits(input, target,pos_weight=pos_weight)
    else:
        return F.binary_cross_entropy(input, target)
    
def dice_loss(input, target):
    input = torch.sigmoid(input)
    input = input.contiguous().view(input.size()[0],-1)
    target = target.contiguous().view(target.size()[0], -1)
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.0001
    c = torch.sum(target * target, 1) + 0.0001
    d = (2*a) / (b+c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def focal_loss(input, target, alpha=1, gamma=2, logits=True, reduce=True):
    if logits:
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
    else:
        bce_loss = F.binary_cross_entropy(input, target, reduce=False)
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt)**gamma * bce_loss
    if reduce:
        return torch.mean(focal_loss)
    else:
        return focal_loss

def focal_loss_v2(input, target, alpha=0.95, gamma=2, size_average=True):
    epsilon = 1e-6
    input = input.contiguous().view(input.size()[0],-1)
    target = target.contiguous().view(target.size()[0], -1)
    pt = torch.sigmoid(input)
    pt = torch.clamp(pt, min=epsilon, max=1-epsilon)
    loss = - alpha * (1 - pt) ** gamma * target * torch.log(pt) - \
           (1 - alpha) * pt ** gamma * (1 - target) * torch.log(1 - pt)
    #pdb.set_trace()
    if size_average:
        loss = torch.mean(loss)
    else:
        loss = torch.sum(loss)
    return loss