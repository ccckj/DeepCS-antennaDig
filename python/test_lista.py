import sys, os
import pdb
import numpy as np 
import torch
import argparse
import models
import time
import copy
import collections
from util.metric import runningScore, nmse_metirc_lista as nmse_metric
from util import AverageMeter, Logger, nmse_loss_v2
from dataset import ListaDataLoader, ArrayResposeLoad
import torch.nn.functional as F
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

def update_score(preds, masks, running_metric, th=0.5):
    preds[preds>=th] = 1
    preds[preds<th] = 0
    preds = preds.astype(np.int32)
    running_metric.update(masks, preds)

def validate(args):
    layer = -1
    avgNmse = AverageMeter()
    running_metric = runningScore(2)

    data_loader = ListaDataLoader(args,flag='val')
    val_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    A = ArrayResposeLoad(measurements=args.measurements, antenna_x=args.antenna_x, antenna_y=args.antenna_y)

    if args.arch == "LISTA":
        model = models.LISTA( A=A, T=args.T, lam=args.lam, untied=args.untied, coord=args.coord)

        
    for param in model.parameters():
        param.requires_grad = False
    device = args.device
    model = model.to(device)
    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
            layer = checkpoint['layer']
            print("Loaded checkpoint '{}' (layer {}|epoch {})".format(args.resume, checkpoint['layer'],checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
        sys.stdout.flush()
    model.eval()

    for idx, (y, x) in enumerate(val_loader):
        print('progress: %d / %d'%(idx+1, len(val_loader)))

        x_hats = model(y.cuda().transpose(0,1))
        nmse = nmse_metric(x_hats[layer], x.cuda().transpose(0,1))
        avgNmse.update(nmse, x.size(0))

        antenna_size = args.antenna_x * args.antenna_y

        x = x.cpu().numpy()
        x_complex = x[:, :antenna_size] + 1j*x[:, antenna_size:]
        x_abs = abs(x_complex)

        mask = x_abs != 0

        x_hat = x_hats[layer].transpose(0,1)
        x_hat = x_hat.cpu().numpy()
        x_complex_hat = x_hat[:, :antenna_size] + 1j*x_hat[:, antenna_size:]
        x_hat_abs = abs(x_complex_hat)

        high_th = np.median(x_hat_abs[mask])
        low_th = np.median(x_hat_abs[1-mask])
        th = (high_th + low_th) / 2
        update_score(x_hat_abs, mask, running_metric, th)


        
    print('F1 score:', running_metric.get_scores())
    print('normalized MSE:', avgNmse.avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',type=str, nargs='?', default='LISTA')
    parser.add_argument('--checkpoint', type=str, default='', metavar='PATH', help='path to save checkpoint')
    parser.add_argument('--batch_size', type=int, nargs='?', default=2000)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3)
    parser.add_argument('--n_epoch', type=int, nargs='?', default=40)
    parser.add_argument('--measurements', type=int, nargs='?', default=256)
    parser.add_argument('--schedule',type=int, nargs='+', default=[10])
    
    parser.add_argument('--T', type=int, nargs='?', default=20, help='Number of layers')
    parser.add_argument('--lam', type=float, nargs='?', default=0.4)
    parser.add_argument('--untied', action="store_true", help="Flag of whether weights are shared within layers.")
    parser.add_argument('--coord',  action='store_true', help="Whether use independent vector thresholds")
    parser.add_argument('--pretrain', type=str, default=None, help="Path to previous saved model to restart")
    parser.add_argument('--resume', type=str, default=None, help='Path to previous saved model to restart from')
    parser.add_argument('--need_validate', type=bool, default=True, help='whether to validate the model after some epochs')
    parser.add_argument('--fault_prob', type=float, help='probability of antenna faulty, it rely on the generated mat')
    parser.add_argument('--SNR', type=int, help="SNR of data")
    parser.add_argument('--sample_nums', type=int, help='total training and validate dataset size')
    parser.add_argument('--antenna_x', type=int, nargs='?', default=16)
    parser.add_argument('--antenna_y', type=int, nargs='?', default=16)
    parser.add_argument('--device', type=str, default='cuda', help="select device to run the model")

    args = parser.parse_args()
    validate(args)