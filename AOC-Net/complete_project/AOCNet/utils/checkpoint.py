import torch
import os
import numpy as np

def load_network_and_optimizer(net, opt, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    opt.load_state_dict(pretrained['optimizer'])
    del(pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove

def load_network_and_not_optimizer(net, opt, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    #opt.load_state_dict(pretrained['optimizer'])
    del(pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove

def load_network(net, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].size()==pretrained_dict[k].size():
            pretrained_dict_update[k] = v
            #if model_dict[k].size()!=pretrained_dict[k].size():
            #    print("ERROR->",k)
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del(pretrained)
    return net.cuda(gpu), pretrained_dict_remove


def load_network_P2T(net, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        print("pretrained_dict_k",k)
        if k in model_dict:
            print("k in new model")
            pretrained_dict_update[k] = v
            #for para in pretrained_dict_update[k].parameters():
            #    para.requires_grad = False
            v.requires_grad = False
        elif k[:7] == 'module.':
            print("k in new model (module)")
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
            #for para in pretrained_dict_update[k].parameters():
            v.requires_grad = False
        else:
            print("k not in new model")
            pretrained_dict_remove.append(k)

    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del(pretrained)
    return net.cuda(gpu), pretrained_dict_remove


def save_network(net, opt, step, save_path, max_keep=8):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save({'state_dict': net.state_dict(), 'optimizer': opt.state_dict()}, save_dir)
    except:
        save_path = './saved_models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save({'state_dict': net.state_dict(), 'optimizer': opt.state_dict()}, save_dir)

    all_ckpt = os.listdir(save_path)
    if len(all_ckpt) > max_keep:
        all_step = []
        for ckpt_name in all_ckpt:
            step = int(ckpt_name.split('_')[-1].split('.')[0])
            all_step.append(step)
        all_step = list(np.sort(all_step))[:-max_keep]
        for step in all_step:
            ckpt_path = os.path.join(save_path, 'save_step_%s.pth' % (step))
            os.system('rm {}'.format(ckpt_path))
