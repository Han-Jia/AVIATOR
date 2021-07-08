import argparse
import os.path as osp
import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
from feat.models.maml_fc_cls import MAML_FC_cls 
from feat.utils import pprint, set_gpu, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval
import shutil
import pdb
def save_model(name):
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))

def generate_task(gmm, way=3, shot=5, query=15):
    class_mean = np.random.permutation(gmm)[:way]
    # print(class_mean)
    task = np.tile(class_mean, (shot+query, 1)) + np.random.randn((shot+query)*way, 2)*0.01
    return torch.autograd.Variable(torch.tensor(task)).float()

def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--way', type=int, default=3)    
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    
    parser.add_argument('--gd_lr', type=float, default=0.01) # lr for gd updates   
    parser.add_argument('--lr', type=float, default=0.001) # lr for meta updates 
    parser.add_argument('--lr_mul', type=float, default=10) # lr is the basic learning rate, while lr * lr_mul is the lr for other parts
    parser.add_argument('--inner_iters', type=int, default=5)
    
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.1)    
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_type', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='toy_cls')       
    parser.add_argument('--init_weights', type=str, default=None) 
    parser.add_argument('--multi_stage', type=bool, default=False)    
    parser.add_argument('--comment', type=str, default='temp') # The temp name to save the reulst file
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gmm_mean_path', type=str, default='./gmm_mean.mat')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type, 'AVIATOR_synthesis', str(args.shot), str(args.way)])
    save_path2 = '_'.join([str(args.gd_lr), str(args.lr), str(args.lr_mul), str(args.inner_iters), 
                           str(args.temperature), str(args.step_size), str(args.gamma), '0.1'] )
    if args.multi_stage:
        save_path2 += '_MS'
    args.save_path = osp.join(save_path1, save_path2)
 

    np.random.seed(0)
    torch.manual_seed(0)
    
    model = MAML_FC_cls(args)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    gmm_mean = sio.loadmat(args.gmm_mean_path)

    val_mean = gmm_mean['val_mean']
    test_mean = gmm_mean['test_mean']

    
    trlog = {}
    trlog['args'] = vars(args)
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

    
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
    


    # basic_inner_step = args.inner_iters
    basic_inner_step = [1, 5, 10]
    test_acc = []
    for iter_mul in range(3):
        # try finetune with different step sizes
        test_acc_record = np.zeros((10000,))
        # args.inner_iters = basic_inner_step * iter_mul   
        args.inner_iters = basic_inner_step[iter_mul] 
        model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
        model.eval()
        model.encoder.is_training = True        

            
        ave_acc = Averager()
        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        for i in range(10000):
            data = generate_task(test_mean)
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query) # KqN x KN x 1
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            # print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))      
            
        m, pm = compute_confidence_interval(test_acc_record)
        test_acc.append((m,pm))
        

    # show results for different iterations
    print('Val Best Epoch {}, Best Val Acc {:.5f}, Val Loss {:.5f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['min_loss']))
    print('Val Best Epoch {}, Best Train Acc {:.5f}, Train Loss {:.5f}'.format(trlog['max_acc_epoch'], trlog['val_train_acc'], trlog['val_train_loss']))
    for i, iter_mul in enumerate([1,2,3]):
        # print('Inner Iter {}, Test Acc {:.5f} + {:.5f}'.format(basic_inner_step * iter_mul, test_acc[i][0], test_acc[i][1]))
        print('Inner Iter {}, Test Acc {:.5f} + {:.5f}'.format(basic_inner_step[i], test_acc[i][0], test_acc[i][1]))
          