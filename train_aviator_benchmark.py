import argparse
import os.path as osp

import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from model.models.mamlp_fc import MAMLP_FC 
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--way', type=int, default=5)    
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    
    parser.add_argument('--gd_lr', type=float, default=0.01) # lr for gd updates   
    parser.add_argument('--lr', type=float, default=0.001) # lr for meta updates 
    parser.add_argument('--lr_mul', type=float, default=10) # lr is the basic learning rate, while lr * lr_mul is the lr for other parts
    parser.add_argument('--inner_iters', type=int, default=1)
    
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB'])    
    # MiniImageNet, ConvNet, './saves/initialization/miniimagenet/con-pre-noaug.pth'
    # MiniImageNet, ResNet, './saves/initialization/miniimagenet/res-pre.pth'
    # CUB, ConvNet, './saves/initialization/cub/con-pre.pth'    
    parser.add_argument('--init_weights', type=str, default=None)
    # parser.add_argument('--init_weights', type=str, default='saves/initialization/miniimagenet/con-pre.pth')      
    # parser.add_argument('--init_weights', type=str, default='saves/initialization/cub/con-pre.pth') 
    parser.add_argument('--comment', type=str, default='temp') # The temp name to save the reulst file
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    if args.init_weights is not None:
        is_pretrain = 'T'
    else:
        is_pretrain = 'F'
    save_path1 = '-'.join([args.dataset, args.model_type, 'AVIATOR_benchmark', str(args.shot), str(args.way)])
    save_path2 = '_'.join([str(args.gd_lr), str(args.lr), str(args.lr_mul), str(args.inner_iters), 
                           str(args.temperature), str(args.step_size), str(args.gamma), is_pretrain])
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    else:
        raise ValueError('Non-supported Dataset.')
    
    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 500, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)  
    
    model = MAMLP_FC(args)
    if args.init_weights is not None:
        # get previous layers
        basic_para = list()
        top_para = list()
        for e in model.named_parameters():
            if 'encoder' not in e[0] or 'FC' in e[0]:
                top_para.append(e[1])
            else:
                basic_para.append(e[1])
        if args.model_type == 'ConvNet':
            optimizer = torch.optim.Adam([{'params': basic_para},
                                         {'params': top_para, 'lr': args.lr * args.lr_mul}], lr=args.lr)            
        elif args.model_type == 'ResNet':
            optimizer = torch.optim.SGD([{'params': basic_para},
                                         {'params': top_para, 'lr': args.lr * args.lr_mul}], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)                
        else:
            raise ValueError('No Such Encoder')
    else:
        if args.model_type == 'ConvNet':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)            
        elif args.model_type == 'ResNet':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)                
        else:
            raise ValueError('No Such Encoder')        
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    
    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    if args.init_weights is not None:
        if args.model_type == 'ConvNet':
            refined_dict = {}
            pretrained_dict = torch.load(args.init_weights)['params']
            for e in pretrained_dict:
                if 'encoder' in e:
                    new_name = e.split('.')
                    new_name = 'encoder.' + '_'.join(new_name[1:3]) + '.' + new_name[-1]
                    refined_dict[new_name] = pretrained_dict[e]
            pretrained_dict = {k: v for k, v in refined_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
        elif args.model_type == 'ResNet':
            pretrained_dict = torch.load(args.init_weights)['params']
            # remove the FC layer
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'FC' not in k}                      
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items() if 'encoder.' + k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
        else:
            raise ValueError('No Such Encoder')
        
        model.load_state_dict(model_dict)    
    
    # record the index of running mean and variance 
    running_dict = {}
    for e in model_dict:
        if 'running' in e:
            key_name = '.'.join(e.split('.')[1:-1])
            if key_name in running_dict:
                continue
            else:
                running_dict[key_name] = {}
            # find the position of BN modules
            component = model.encoder
            for att in key_name.split('.'):
                if att.isdigit():
                    component = component[int(att)]
                else:
                    component = getattr(component, att)
            
            running_dict[key_name]['mean'] = component.running_mean
            running_dict[key_name]['var'] = component.running_var
                
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['val_train_acc'] = 0.0
    trlog['min_loss'] = 10000
    trlog['val_train_loss'] = 10000
    trlog['max_acc_epoch'] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(log_dir=args.save_path)
    
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
            
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()
        model.train()
        tl = Averager()
        ta = Averager()
            
        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query) # KqN x KN x 1
            # compute loss
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            if (i-1) % 50 == 0:
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                      .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()
        vl = Averager()
        va = Averager()
            
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        model.eval()
        model.encoder.is_training = True
        # record the runing mean and variance before validation
        for e in running_dict:
            running_dict[e]['mean_copy'] = deepcopy(running_dict[e]['mean'])
            running_dict[e]['var_copy'] = deepcopy(running_dict[e]['var'])
        
        for i, batch in enumerate(val_loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query) # KqN x KN x 1
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)
            
            # reset the running mean and variance
            for e in running_dict:
                running_dict[e]['mean'] = deepcopy(running_dict[e]['mean_copy'])
                running_dict[e]['var'] = deepcopy(running_dict[e]['mean_copy'])

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)    
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['min_loss'] = vl
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        save_model('epoch-last')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()

    # Test Phase
    # compute the training loss    
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    val_train_sampler = CategoriesSampler(trainset.label, 500, args.way, args.shot + args.query)
    val_train_loader = DataLoader(dataset=trainset, batch_sampler=val_train_sampler, num_workers=8, pin_memory=True)  
    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()
    model.encoder.is_training = True    
    vtl = Averager()
    vta = Averager()            
    # record the runing mean and variance before validation
    for e in running_dict:
        running_dict[e]['mean_copy'] = deepcopy(running_dict[e]['mean'])
        running_dict[e]['var_copy'] = deepcopy(running_dict[e]['var'])
    
    for i, batch in enumerate(val_train_loader, 1):
        if torch.cuda.is_available():
            data, _ = [_.cuda() for _ in batch]
        else:
            data = batch[0]
        p = args.shot * args.way
        data_shot, data_query = data[:p], data[p:]
        logits = model(data_shot, data_query) # KqN x KN x 1
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)
        vtl.add(loss.item())
        vta.add(acc)
        # reset the running mean and variance
        for e in running_dict:
            running_dict[e]['mean'] = deepcopy(running_dict[e]['mean_copy'])
            running_dict[e]['var'] = deepcopy(running_dict[e]['mean_copy'])

    vtl = vtl.item()
    vta = vta.item()
    print('val_train, loss={:.4f} acc={:.4f}'.format(vtl, vta))
    trlog['val_train_acc'] = vta
    trlog['val_train_loss'] = vtl
    
    # test phase
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    basic_inner_step = args.inner_iters
    test_acc = []
    for iter_mul in [1,2,3]:
        # try finetune with different step sizes
        test_acc_record = np.zeros((10000,))
        args.inner_iters = basic_inner_step * iter_mul    
        model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
        model.eval()
        model.encoder.is_training = True        
        # record the runing mean and variance before validation
        for e in running_dict:
            running_dict[e]['mean_copy'] = deepcopy(running_dict[e]['mean'])
            running_dict[e]['var_copy'] = deepcopy(running_dict[e]['var'])
            
        ave_acc = Averager()
        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query) # KqN x KN x 1
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
            
            # reset the running mean and variance
            for e in running_dict:
                running_dict[e]['mean'] = deepcopy(running_dict[e]['mean_copy'])
                running_dict[e]['var'] = deepcopy(running_dict[e]['mean_copy'])         
            
        m, pm = compute_confidence_interval(test_acc_record)
        test_acc.append((m,pm))
        

    # show results for different iterations
    print('Val Best Epoch {}, Best Val Acc {:.5f}, Val Loss {:.5f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['min_loss']))
    print('Val Best Epoch {}, Best Train Acc {:.5f}, Train Loss {:.5f}'.format(trlog['max_acc_epoch'], trlog['val_train_acc'], trlog['val_train_loss']))
    for i, iter_mul in enumerate([1,2,3]):
        print('Inner Iter {}, Test Acc {:.5f} + {:.5f}'.format(basic_inner_step * iter_mul, test_acc[i][0], test_acc[i][1]))
        
    import pdb
    pdb.set_trace()
    
    with open(args.comment+'.txt', 'w') as f:
        # save temp results
        f.write(','.join([str(trlog['max_acc_epoch']), str(trlog['max_acc']), str(trlog['min_loss']), 
                          str(trlog['val_train_acc']), str(trlog['val_train_loss']),
                          '{:.5f} + {:.5f}'.format(test_acc[0][0], test_acc[0][1]),
                          '{:.5f} + {:.5f}'.format(test_acc[1][0], test_acc[1][1]),
                          '{:.5f} + {:.5f}'.format(test_acc[2][0], test_acc[2][1])]))    