import torch
import torch.nn.functional as F
import math
from torch import nn
import numpy as np

class IA_gate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1. + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x

class IA_gate_dp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate_dp, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(0.6)
    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1. + torch.tanh(self.dp(a))
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x

class IA_gate_dp_vis(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate_dp_vis, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(0.6)
    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        weight =  torch.tanh(a)
        delta = weight.unsqueeze(-1).unsqueeze(-1)*x
        a = 1. + torch.tanh(self.dp(a))
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        
        return x,delta

class IA_gate_2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate_2, self).__init__()
        self.IA1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(True)
        self.IA2 = nn.Linear(out_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA1(IA_head)
        a = self.relu(a)
        a = self.IA2(a)
        a = 1. + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x

class IA_gate_2_dp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate_2_dp, self).__init__()
        self.IA1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(True)
        self.IA2 = nn.Linear(out_dim, out_dim)
        self.dp1 = nn.Dropout(0.6)
        self.dp2 = nn.Dropout(0.6)

    def forward(self, x, IA_head):
        a = self.IA1(IA_head)
        a = self.relu(self.dp1(a))
        a = self.IA2(a)
        a = 1. + torch.tanh(self.dp2(a))
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x
def calculate_attention_head(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):

    ref_head = ref_embedding * ref_label
    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)

    return total_head

def calculate_attention_head_for_eval(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num
    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    return total_head


def calculate_attention_head_p_m(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):

    ref_head = ref_embedding * ref_label
    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    return total_head,ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg

def calculate_attention_head_for_eval_p_m(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num
    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    ref_head = torch.cat([ref_head_pos, ref_head_neg], dim=1)
    prev_head = torch.cat([prev_head_pos, prev_head_neg], dim=1)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    return total_head,ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg

class InteractModule(nn.Module):
    def __init__(self, planes):
        super(InteractModule, self).__init__()
        self.conv = nn.Conv2d(planes*2, planes*2, kernel_size=3, bias=False)
        self.bn = nn.GroupNorm(32, planes*2)

    def forward(self, feat1,feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x =  self.conv (x)
        #print("InteractModule_conv_size",x.size())
        x = self.bn(x)
        #print("InteractModule_bn_size",x.size())

        x = 1. + torch.tanh(x)
        return x

def LocationEmbedding(f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    position_mat = torch.cat((cx, cy, w, h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, -1)
    position_mat = position_mat.view(f_g.shape[0], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(f_g.shape[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding

def get_xyxy_from_mask( mask):
    uMax = mask.shape[-1]
    vMax = mask.shape[-2]
    #print(mask.shape)
    #print(uMax,vMax)
    #print(mask.sum())
    #print(np.nonzero(mask))
    if mask.sum()<1e-6:
        return [0,0,1,1]
    
    vs, us = np.nonzero(mask)
    
    y0, y1 = vs.min(), vs.max()
    x0, x1 = us.min(), us.max()
    return [x0/uMax, y0/vMax, x1/uMax, y1/vMax]



def locational_embedding_aug_v2(data,ref_emb,spatial_dim):
    pos_emb_all=[]
    for i in range(data.size()[0]):
        pos_emb=[]
        for j in range(data.size()[1]):
            mask=data[i,j,:,:]
            #print(mask.size())
            xyxy_mask = np.array([get_xyxy_from_mask(np.array(mask.cpu()))], dtype=np.float32)
            embeds = LocationEmbedding(torch.from_numpy(xyxy_mask).to(device=ref_emb.device) ).squeeze(0).cpu()
            pos_emb.append(embeds.numpy())
        pos_emb_all.append(pos_emb)
    pos_emb_all_np=np.array(pos_emb_all).astype(np.float64)
    #print(pos_emb_all_np.shape)
    pos_emb_all_ts=torch.from_numpy(pos_emb_all_np).squeeze(1).to(device=ref_emb.device)
    #print(pos_emb_all_ts.size())
    return  pos_emb_all_ts.float()



def locational_embedding_aug(data,ref_emb,spatial_dim): # data==label,emb=(3,100)
    #data = torch.from_numpy(np.random.rand(3, 1, 121, 229))
    #emb = torch.from_numpy(np.random.rand(3, 100))
    #print(data.size())
    pixel_num=data.size()[-1]*data.size()[-2]
    data_x = torch.sum(data, dim=(2)).to(device=ref_emb.device)
    data_y = torch.sum(data, dim=(3)).to(device=ref_emb.device)

    pos_emb_all=[]
    for i in range(data_x.size()[0]):
        pos_emb=[]
        for j in range(data_x.size()[1]):
            data_x_cur = data_x[i,j,:]
            data_y_cur = data_y[i,j,:]

            data_x_points = (data_x_cur>0).nonzero()
            data_y_points = (data_y_cur>0).nonzero()
            
            if (data_x_points.size()[0])==0:
                x_min=0
                y_min=0
                x_max= data.size()[-2]
                y_max =data.size()[-1]
            else:
                x_min = (data_x_points.min())
                x_max = (data_x_points.max())
                y_min = (data_y_points.min())
                y_max = (data_y_points.max())

            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5
            w = (x_max - x_min) + 1.
            h = (y_max - y_min) + 1.

            cur_emb = [cx/data.size()[-2],cy/data.size()[-1],w/data.size()[-2],h/data.size()[-1]] 
            
            data_x_cur_bin=np.array_split((data_x_cur.cpu()),spatial_dim)
            data_y_cur_bin=np.array_split((data_y_cur.cpu()),spatial_dim)
            for data_i in (data_x_cur_bin):
                pos_x_emb=(torch.sum(data_i.cpu()).cpu().unsqueeze(-1)).cpu()/pixel_num
                cur_emb.append(pos_x_emb)
            for data_i in (data_y_cur_bin):
                pos_y_emb=(torch.sum(data_i.cpu()).cpu().unsqueeze(-1)).cpu()/pixel_num
                cur_emb.append(pos_y_emb)
                
                
            pos_emb.append(cur_emb)
            
        pos_emb_all.append(pos_emb)
        
    pos_emb_all_ts=torch.from_numpy(np.array(pos_emb_all).astype(np.float64)).squeeze(1).to(device=ref_emb.device)


    return  pos_emb_all_ts.float() # cx, cy, w, h, x_bin-117, y_bin-117






"""
def locational_embedding_aug(data,ref_emb,spatial_dim): # data==label,emb=(3,100)
    #data = torch.from_numpy(np.random.rand(3, 1, 121, 229))
    #emb = torch.from_numpy(np.random.rand(3, 100))
    #print(data.size())
    pixel_num=torch.sum(data, dim=(2,3))
    data_x = torch.sum(data, dim=(2)).to(device=ref_emb.device)
    data_y = torch.sum(data, dim=(3)).to(device=ref_emb.device)

    pos_emb_all=[]
    for i in range(data_x.size()[0]):
        pos_emb=[]
        for j in range(data_x.size()[1]):
        
            data_x_cur = data_x[i,j,:]
            data_y_cur = data_y[i,j,:]
            data_x_points = (data_x_cur>0).nonzero()
            data_y_points = (data_y_cur>0).nonzero()
            if (data_x_points.size()[0])==0:
                x_min=0
                y_min=0
                x_max= data.size()[-2]
                y_max =data.size()[-1]
            else:
                x_min = (data_x_points.min())
                x_max = (data_x_points.max())
                y_min = (data_y_points.min())
                y_max = (data_y_points.max())

            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5
            w = (x_max - x_min) + 1.
            h = (y_max - y_min) + 1.

            cur_emb = [cx,cy,w,h]
        
            pos_emb.append(cur_emb)
        pos_emb_all.append(pos_emb)
    
    emb=torch.from_numpy(np.array(pos_emb_all).astype(np.uint8)).squeeze(1).to(device=ref_emb.device).byte()
    epsilon=1e-6

    for i, data_i in enumerate(data_x.chunk(spatial_dim,2)):
        pos_x_emb=(torch.sum(data_i,dim=(1,2))).unsqueeze(-1).to(device=ref_emb.device)
        #print(emb.device)
        #print(pos_x_emb.device)
        emb=torch.cat([emb,pos_x_emb.byte()/(pixel_num.byte())],dim=(1))
    for i, data_i in enumerate(data_y.chunk(spatial_dim,2)):
        pos_y_emb=(torch.sum(data_i,dim=(1,2))).unsqueeze(-1).to(device=ref_emb.device)
        emb=torch.cat([emb,pos_y_emb.byte()/(pixel_num.byte())],dim=(1))

    #print(emb)
    return emb
"""


def calculate_attention_head_GL(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):

    ref_head = ref_embedding * ref_label

    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head_G = torch.cat([ref_head_pos, ref_head_neg], dim=1)
    total_head_L = torch.cat([prev_head_pos, prev_head_neg], dim=1)
    return total_head_G,total_head_L

def calculate_attention_head_interact(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):

    ref_head = ref_embedding * ref_label

    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head_fg = torch.cat([ref_head_pos,  prev_head_pos], dim=1)
    total_head_bg = torch.cat([ ref_head_neg,  prev_head_neg], dim=1)
    return total_head_fg,total_head_bg



def calculate_attention_head_spatial_aug(ref_embedding, ref_label, prev_embedding, prev_label, epsilon,spatial_dim):

    ref_head = ref_embedding * ref_label

    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    prev_loc_emb1 = locational_embedding_aug(prev_label,ref_embedding,spatial_dim)
    prev_loc_emb2 = locational_embedding_aug_v2(prev_label,ref_embedding,spatial_dim)
    total_head = torch.cat([total_head,prev_loc_emb1.float(),prev_loc_emb2.float()], dim=1)
    
    return total_head



def calculate_attention_head_spatial_only(ref_embedding, ref_label, prev_embedding, prev_label, epsilon,spatial_dim):
    prev_loc_emb1 = locational_embedding_aug(prev_label,ref_embedding,spatial_dim)
    prev_loc_emb2 = locational_embedding_aug_v2(prev_label,ref_embedding,spatial_dim)
    total_head = torch.cat([prev_loc_emb1.float(),prev_loc_emb2.float()], dim=1)
    
    return total_head

def calculate_attention_head_spatial_aug2(ref_embedding, ref_label, prev_embedding, prev_label, epsilon,spatial_dim):

    ref_head = ref_embedding * ref_label

    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    prev_loc_emb1 = locational_embedding_aug(prev_label,ref_embedding,spatial_dim)
    #prev_loc_emb2 = locational_embedding_aug_v2(prev_label,ref_embedding,spatial_dim)
    total_head = torch.cat([total_head,prev_loc_emb1.float()], dim=1)
    
    return total_head


def calculate_attention_head_GL_for_eval(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        

        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num




    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + (epsilon))
    prev_head_neg = prev_head_neg / (prev_neg_num + (epsilon))
    #print("prev_head.size()",prev_head.size())
    #print("prev_head_pos.size()",prev_head_pos.size())
    
    #total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    total_head_G = torch.cat([ref_head_pos, ref_head_neg], dim=1)
    total_head_L = torch.cat([prev_head_pos, prev_head_neg], dim=1)
    return total_head_G,total_head_L

def calculate_attention_head_for_eval_interact(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        

        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num




    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + (epsilon))
    prev_head_neg = prev_head_neg / (prev_neg_num + (epsilon))
    #print("prev_head.size()",prev_head.size())
    #print("prev_head_pos.size()",prev_head_pos.size())
    
    total_head_fg = torch.cat([ref_head_pos,  prev_head_pos], dim=1)
    total_head_bg = torch.cat([ ref_head_neg,  prev_head_neg], dim=1)
    #print("pre_total_head_size，",total_head.size())
    """
    for idx in range(len(ref_embeddings)):
        ref_loc_emb = locational_embedding_aug(ref_label,ref_embedding)
        total_head = torch.cat([total_head,ref_loc_emb], dim=1)
    """
    #prev_loc_emb = locational_embedding_aug(prev_label,ref_embedding)
    #total_head = torch.cat([total_head,prev_loc_emb], dim=1)
    #print("post_total_head_size，",total_head.size())

    return total_head_fg,total_head_bg

def calculate_attention_head_for_eval_spatial_aug(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon,spatial_dim):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
       
        

        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num




    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)
    #print("prev_head.size()",prev_head.size())
    #print("prev_head_pos.size()",prev_head_pos.size())
    
    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    #print("pre_total_head_size，",total_head.size())
    """
    for idx in range(len(ref_embeddings)):
        ref_loc_emb = locational_embedding_aug(ref_label,ref_embedding)
        total_head = torch.cat([total_head,ref_loc_emb], dim=1)
    """
    prev_loc_emb1 = locational_embedding_aug(prev_label,ref_embedding,spatial_dim)
    prev_loc_emb2 = locational_embedding_aug_v2(prev_label,ref_embedding,spatial_dim)
    total_head = torch.cat([total_head,prev_loc_emb1.float(),prev_loc_emb2.float()], dim=1)
    #print("post_total_head_size，",total_head.size())

    return total_head

def calculate_attention_head_for_eval_spatial_only(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon,spatial_dim):
    prev_loc_emb1 = locational_embedding_aug(prev_label,ref_embedding,spatial_dim)
    prev_loc_emb2 = locational_embedding_aug_v2(prev_label,ref_embedding,spatial_dim)
    total_head = torch.cat([prev_loc_emb1.float(),prev_loc_emb2.float()], dim=1)
    #print("post_total_head_size，",total_head.size())

    return total_head

def calculate_attention_head_for_eval_spatial_aug2(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon,spatial_dim):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
       
        

        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num




    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)
    #print("prev_head.size()",prev_head.size())
    #print("prev_head_pos.size()",prev_head_pos.size())
    
    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    #print("pre_total_head_size，",total_head.size())
    """
    for idx in range(len(ref_embeddings)):
        ref_loc_emb = locational_embedding_aug(ref_label,ref_embedding)
        total_head = torch.cat([total_head,ref_loc_emb], dim=1)
    """
    prev_loc_emb1 = locational_embedding_aug(prev_label,ref_embedding,spatial_dim)
    #prev_loc_emb2 = locational_embedding_aug_v2(prev_label,ref_embedding,spatial_dim)
    total_head = torch.cat([total_head,prev_loc_emb1.float()], dim=1)
    #print("post_total_head_size，",total_head.size())

    return total_head




