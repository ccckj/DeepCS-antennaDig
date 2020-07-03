import sys, os
import pdb
import numpy as np 
import torch
import argparse
import models
import time
from util.metric import runningScore, nmse_metirc_lista as nmse_metric
from util import AverageMeter, Logger, nmse_loss_v2
from dataset import ListaDataLoader, ArrayResposeLoad
import torch.nn.functional as F
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR


def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint.pth.rar'):
    filename = str(epoch) + '__' + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def save_best_checkpoint(state, checkpoint='checkpoint', filename='bestCheckpoint.pth.rar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    for batch_idx, (y, x) in enumerate(train_loader):
        end = time.time()
        y = y.cuda()
        x = x.cuda()
        y = torch.transpose(y, 0, 1)
        x = torch.transpose(x, 0, 1)
        x_hats = model(y)

        loss = nmse_loss_v2(x_hats[-1], x)

        losses.update(loss.item(), y.size(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if batch_idx % 10 == 0:
            output_log = '({batch}/{size} Batch: {bt:.3f}s | TOTAL: {total:.3f}min | ETA: {eta:.3f}min | Loss:{nmse:.6f} )'.format(
                batch = batch_idx + 1,
                size = len(train_loader),
                bt = batch_time.avg,
                total = batch_time.avg * (len(train_loader)) / 60.0,
                eta = batch_time.avg * (len(train_loader) - batch_idx) /60.0,
                nmse = losses.avg
            )
            print(output_log)
            sys.stdout.flush()
    return losses.avg

def validate(model, args):
    data_loader = ListaDataLoader(args,flag='val')
    val_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    avgNmse = AverageMeter()

    #todo define F1score
    model.eval()
    for idx, (y, x) in enumerate(val_loader):
        y = y.cuda()
        x = x.cuda()
        y = torch.transpose(y, 0, 1)
        x = torch.transpose(x, 0, 1)
        x_hats = model(y)
        nmse = nmse_metric(x_hats[-1], x)
        avgNmse.update(nmse, x.size(-1))
        
    return avgNmse.avg

def main(args):
    start_epoch = 0
    if args.checkpoint=='':
        args.checkpoint = "finetune_lista_checkpoint/n%d_s%d_p%d_snr%d/%s_bs_%d_ep_%d/measurements%d"\
        %(args.sample_nums, args.antenna_x*args.antenna_y, args.fault_prob*100, args.SNR, args.arch, args.batch_size, args.n_epoch, args.measurements)
    print('checkpoint path: %s'%args.checkpoint)
    print('init lr: %.8f'%args.lr)
    #print('schedule: ', args.schedule)

    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    data_loader = ListaDataLoader(args)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    # sensing matrix
    A = ArrayResposeLoad(measurements=args.measurements, antenna_x=args.antenna_x, antenna_y=args.antenna_y)

    if args.arch == "LISTA":
        model = models.LISTA( A=A, T=args.T, lam=args.lam, untied=args.untied, coord=args.coord)

    model = torch.nn.DataParallel(model).cuda()
    #for p,v in model.named_parameters():
    #    pdb.set_trace()

    #if hasattr(model.module, 'optimizer'):
    #    optimizer = model.module.optimizer
    #else:
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'nmse'])
    elif args.resume:
        print("Resuming from checkpoint")
        assert os.path.isfile(args.resume), 'Error: no resume checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        #logger.set_names(['Epoch', 'Learning Rate', 'Train Loss'])
        logger.set_names(['Epoch', 'Learning Rate', 'nmse'])

    bestResult = np.inf
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=70, eta_min=5e-6)
    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f'%(epoch+1, args.n_epoch, optimizer.param_groups[0]['lr']))

        nmse = train(train_loader, model, optimizer, epoch)
        lr_scheduler.step()
        #save_checkpoint({
        #    'epoch':epoch+1,
        #    'state_dict': model.state_dict(),
        #    'lr': args.lr, 
        #    'optimizer': optimizer.state_dict(),
        #    }, epoch+1, checkpoint=args.checkpoint)

        if args.need_validate and (epoch+1) % 5 == 0:
            print('Validating the model')
            avgNmse= validate(model, args)
            print('The normalized mse in val set is:{nmse:.6f}'.format(nmse=avgNmse))

            if True and avgNmse < bestResult:
                print('Save the best model!')
                bestResult = avgNmse
                save_best_checkpoint({
                    'epoch':epoch+1,
                    'state_dict': model.state_dict(),
                    'lr': args.lr,
                    'optimizer': optimizer.state_dict(),
                    }, checkpoint=args.checkpoint)

        logger.append([epoch+1, optimizer.param_groups[0]['lr'], nmse])
    logger.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',type=str, nargs='?', default='LISTA')
    parser.add_argument('--checkpoint', type=str, default='', metavar='PATH', help='path to save checkpoint')
    parser.add_argument('--batch_size', type=int, nargs='?', default=2000)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3)
    parser.add_argument('--n_epoch', type=int, nargs='?', default=350)
    parser.add_argument('--measurements', type=int, nargs='?', default=256)
    parser.add_argument('--schedule',type=int, nargs='+', default=[200])
    
    parser.add_argument('--T', type=int, nargs='?', default=20, help='Number of layers')
    parser.add_argument('--lam', type=float, nargs='?', default=0.4)
    parser.add_argument('--untied', action="store_true", help="Flag of whether weights are shared within layers.")
    parser.add_argument('--coord', action="store_true", help="Whether use independent vector thresholds")
    parser.add_argument('--pretrain', type=str, default=None, help="Path to previous saved model to restart")
    parser.add_argument('--resume', type=str, default=None, help='Path to previous saved model to restart from')
    parser.add_argument('--need_validate', type=bool, default=True, help='whether to validate the model after some epochs')
    parser.add_argument('--fault_prob', type=float, help='probability of antenna faulty, it rely on the generated mat')
    parser.add_argument('--SNR', type=int, help="SNR of data")
    parser.add_argument('--sample_nums', type=int, help='total training and validate dataset size')
    parser.add_argument('--antenna_x', type=int, nargs='?', default=16)
    parser.add_argument('--antenna_y', type=int, nargs='?', default=16)

    args = parser.parse_args()
    print(args)
    main(args)