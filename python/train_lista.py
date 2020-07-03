import sys, os
import pdb
import numpy as np 
import torch
import argparse
import models
import time
import copy
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

def save_checkpoint(state, layer, epoch, checkpoint='checkpoint', filename='checkpoint.pth.rar'):
    filename = 'layer_{layer}_epoch_{epoch}_'.format(layer=layer, epoch=epoch) + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def save_best_checkpoint(state, checkpoint='checkpoint', filename='bestCheckpoint.pth.rar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def train(train_loader, model, optimizer, epoch, layer):
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

        loss = nmse_loss_v2(x_hats[layer], x)

        losses.update(loss.item(), y.size(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if batch_idx % 10 == 0:
            output_log = '(Layer: {layer} | {batch}/{size} Batch: {bt:.3f}s | TOTAL: {total:.3f}min | ETA: {eta:.3f}min | Loss:{nmse:.6f} )'.format(
                layer = layer,
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

def validate(model, args, layer):
    data_loader = ListaDataLoader(args,flag='val')
    val_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1000,
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
        nmse = nmse_metric(x_hats[layer], x)
        avgNmse.update(nmse, x.size(-1))
        
    return avgNmse.avg

def finetune(args, model, train_loader, start_epoch, logger, bestResult):
    print('\nStarting finetune stage!!\n')
    print('finetune learning rate is:',args.ft_lr)
    layer = args.T 

    if start_epoch < args.n_epoch:
        start_epoch = args.n_epoch + 1

    for para in model.parameters():
        para.requires_grad = True
    #if args.resume is None:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ft_lr)

    for epoch in range(start_epoch, args.ft_epoch+args.n_epoch):
        print('\nEpoch: [%d | %d] LR: %f'%(epoch+1, args.ft_epoch, optimizer.param_groups[0]['lr']))
        nmse = train(train_loader, model, optimizer, epoch, layer)
        save_checkpoint({
                'layer': layer,
                'epoch':epoch,
                'state_dict': model.state_dict(),
                'lr': args.lr, 
                'lr_scheduler':{},
                'optimizer': optimizer.state_dict(),
                }, layer,epoch+1, checkpoint=args.checkpoint)
        if args.need_validate and (epoch+1) % 5 == 0:
            print('Validating the model')
            avgNmse= validate(model, args, layer)
            print('The normalized mse in val set is:{nmse:.6f}'.format(nmse=avgNmse))

            if True and avgNmse < bestResult:
                print('Save the best model!')
                bestResult = avgNmse
                save_best_checkpoint({
                    'layer':layer,
                    'epoch':epoch,
                    'state_dict': model.state_dict(),
                    'lr': args.lr,
                    'lr_scheduler':{},
                    'optimizer': optimizer.state_dict(),
                    }, checkpoint=args.checkpoint)

        logger.append([layer, epoch+1, optimizer.param_groups[0]['lr'], nmse])



def main(args):
    start_epoch = 0
    start_layer = 1
    if args.checkpoint=='':
        args.checkpoint = "lista_checkpoint/n%d_s%d_p%d_snr%d/%s_bs_%d_ep_%d/measurements%d"\
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

    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        d = collections.OrderedDict()
        keys = list(checkpoint['state_dict'].keys())
        for pname, para in model.named_parameters():
            if pname in keys and checkpoint['state_dict'][pname].shape == para.shape:
                d[pname] = checkpoint['state_dict'][pname]
        model.load_state_dict(d)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Layer', 'Epoch', 'Learning Rate', 'nmse'])
    elif args.resume:
        print("Resuming from checkpoint")
        assert os.path.isfile(args.resume), 'Error: no resume checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        start_layer = checkpoint['layer']
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Layer', 'Epoch', 'Learning Rate', 'nmse'])
    
    bestResult = np.inf
    for layer in range(start_layer, args.T+1):
        print('Start training layer:{}'.format(layer))
        if args.untied:
            for name, para in model.named_parameters():
                if name.endswith('_{}'.format(layer)):
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        else:
            for name, para in model.named_parameters():
                if name.endswith('W') or name.endswith('B'):
                    para.requires_grad = True
                    continue
                if name.endswith('theta_{}'.format(layer)):
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        #for name, para in model.named_parameters():
        #    pdb.set_trace()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad==True, model.parameters()), lr=args.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=5e-6)

        if args.resume:
            checkpoint = torch.load(args.resume)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        for epoch in range(start_epoch, args.n_epoch):
            start_epoch = 0
            #adjust_learning_rate(args, optimizer, epoch)
            print('\nEpoch: [%d | %d] LR: %f'%(epoch+1, args.n_epoch, optimizer.param_groups[0]['lr']))

            nmse = train(train_loader, model, optimizer, epoch, layer)
        
            lr_scheduler.step()

            save_checkpoint({
                'layer': layer,
                'epoch':epoch,
                'state_dict': model.state_dict(),
                'lr': args.lr, 
                'lr_scheduler':lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, layer,epoch+1, checkpoint=args.checkpoint)

            if args.need_validate and (epoch+1) % 5 == 0:
                print('Validating the model')
                avgNmse= validate(model, args, layer)
                print('The normalized mse in val set is:{nmse:.6f}'.format(nmse=avgNmse))

                if True and avgNmse < bestResult:
                    print('Save the best model!')
                    bestResult = avgNmse
                    save_best_checkpoint({
                        'layer':layer,
                        'epoch':epoch,
                        'state_dict': model.state_dict(),
                        'lr': args.lr,
                        'lr_scheduler':lr_scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, checkpoint=args.checkpoint)

            logger.append([layer, epoch+1, optimizer.param_groups[0]['lr'], nmse])
        #recovery parameters
        
        args.resume = None

    # lastly finetune the model
    finetune(args, model, train_loader, start_epoch, logger, bestResult)
    logger.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',type=str, nargs='?', default='LISTA')
    parser.add_argument('--checkpoint', type=str, default='', metavar='PATH', help='path to save checkpoint')
    parser.add_argument('--batch_size', type=int, nargs='?', default=2000)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3, help='learning rate in layer training stage')
    parser.add_argument('--ft_lr', type=float, nargs='?', default=1e-3, help='learning rate in finetune stage')
    parser.add_argument('--n_epoch', type=int, nargs='?', default=50)
    parser.add_argument('--ft_epoch', type=int, nargs='?', default=300)
    #parser.add_argument('--schedule',type=int, nargs='+', default=[10])
    parser.add_argument('--T', type=int, nargs='?', default=20, help='Number of layers')
    parser.add_argument('--lam', type=float, nargs='?', default=0.4)
    parser.add_argument('--untied', action="store_true", help="Flag of whether weights are shared within layers.")
    parser.add_argument('--coord',  action='store_true', help="Whether use independent vector thresholds")
    parser.add_argument('--pretrain', type=str, default=None, help="Path to previous saved model to restart")
    parser.add_argument('--resume', type=str, default=None, help='Path to previous saved model to restart from')
    parser.add_argument('--need_validate', type=bool, default=True, help='whether to validate the model after some epochs')
    parser.add_argument('--measurements', type=int, nargs='?', default=256)
    parser.add_argument('--fault_prob', type=float, help='probability of antenna faulty, it rely on the generated mat')
    parser.add_argument('--SNR', type=int, help="SNR of data")
    parser.add_argument('--sample_nums', type=int, help='total training and validate dataset size')
    parser.add_argument('--antenna_x', type=int, nargs='?', default=16)
    parser.add_argument('--antenna_y', type=int, nargs='?', default=16)

    args = parser.parse_args()
    print(args)
    main(args)