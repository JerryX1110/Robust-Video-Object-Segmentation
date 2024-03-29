import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.attention import IA_gate
from networks.layers.gct import Bottleneck, GCT
from networks.layers.aspp import ASPP
from networks.p2t.conditioning_layer import conditioning_layer, conditioning_block


class CalibrationDecoding(nn.Module):
    def __init__(self,
            in_dim=256,
            attention_dim=400,
            embed_dim=100,
            refine_dim=48,
            low_level_dim=256,
            beta_percentage=0.3):
        super(CalibrationDecoding,self).__init__()
        self.embed_dim = embed_dim
        IA_in_dim = attention_dim
        self.unc_topk_ratio = unc_topk_ratio
        self.IA1 = IA_gate(IA_in_dim, in_dim)
        self.layer1 = Bottleneck(in_dim, embed_dim)

        
        self.layer2 = Bottleneck(embed_dim, embed_dim, 1, 2)
        self.CLB2 = conditioning_block(
                in_dim=embed_dim,
                attention_dim=IA_in_dim,
            beta_percentage=self.beta_percentage)

        
        self.layer3 = Bottleneck(embed_dim, embed_dim * 2, 2)
        self.CLB3 = conditioning_block(
                in_dim=embed_dim,
                attention_dim=IA_in_dim,
            beta_percentage=self.beta_percentage)

        
        self.layer4 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 2)
        self.CLB4 = conditioning_block(
                in_dim=embed_dim*2,
                attention_dim=IA_in_dim,
            beta_percentage=self.beta_percentage)
        
        self.layer5 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)
        self.CLB5 = conditioning_block(
                in_dim=embed_dim*2,
                attention_dim=IA_in_dim,
            beta_percentage=self.beta_percentage)
        
        self.IA9 = IA_gate(IA_in_dim+embed_dim * 2, embed_dim * 2)
        self.ASPP = ASPP()

        self.M1_Reweight_Layer_1 = IA_gate(IA_in_dim, embed_dim * 2)
        self.M1_Bottleneck_1 = Bottleneck(embed_dim*2, embed_dim * 2, 1)

        self.M1_Reweight_Layer_2 = IA_gate(IA_in_dim, embed_dim * 2)
        self.M1_Bottleneck_2 = Bottleneck(embed_dim*2, embed_dim * 1, 1)
        
        self.M1_Reweight_Layer_3 = IA_gate(IA_in_dim, embed_dim * 1)
        self.M1_Bottleneck_3 = Bottleneck(embed_dim*1, embed_dim * 1, 1)

        self.M2_Reweight_Layer_1 = IA_gate(IA_in_dim, embed_dim * 2)
        self.M2_Bottleneck_1 = Bottleneck(embed_dim*2, embed_dim * 2, 1)
        
        self.M2_Reweight_Layer_2 = IA_gate(IA_in_dim, embed_dim * 2)
        self.M2_Bottleneck_2 = Bottleneck(embed_dim*2, embed_dim * 1, 1)
        
        self.M2_Reweight_Layer_3 = IA_gate(IA_in_dim, embed_dim *1)
        self.M2_Bottleneck_3 = Bottleneck(embed_dim*1, embed_dim * 1, 1)


        self.GCT_sc = GCT(low_level_dim + embed_dim)
        self.conv_sc = nn.Conv2d(low_level_dim + embed_dim, refine_dim, 1, bias=False)
        self.bn_sc = nn.GroupNorm(int(refine_dim / 4), refine_dim)
        self.relu = nn.ReLU(inplace=True)

        self.IA10 = IA_gate(IA_in_dim+embed_dim + refine_dim, embed_dim + refine_dim)
        self.conv1 = nn.Conv2d(embed_dim + refine_dim, int(embed_dim / 2), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, int(embed_dim / 2))


        self.IA11 = IA_gate(IA_in_dim+int(embed_dim / 2), int(embed_dim / 2))
        self.conv2 = nn.Conv2d(int(embed_dim / 2), int(embed_dim / 2), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, int(embed_dim / 2))

        self.IA_final_fg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)
        self.IA_final_bg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)

        nn.init.kaiming_normal_(self.conv_sc.weight,mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out', nonlinearity='relu')


    def forward(self, x, IA_head=None,memory_list=None,low_level_feat=None,to_cat_previous_frame=None):


        x = self.IA1(x, IA_head) 
        x = self.layer1(x)

        # start: Calibration #1
        x = self.CLB2(x,IA_head)
        # end: Calibration #1
        
        x = self.layer2(x)

        # start: Calibration #2
        x = self.CLB3(x,IA_head)
        # end: Calibration #2
        
        x = self.layer3(x)

        # start: Calibration #3
        x = self.CLB4(x,IA_head)
        # end: Calibration #3
        
        x = self.layer4(x)

        # start: Calibration #4)
        x = self.CLB5(x,IA_head)
        # end: Calibration #4
        
        x = self.layer5(x)

        px1 = torch.nn.functional.avg_pool2d(x,kernel_size=(x.size()[-2],x.size()[-1]),padding = 0)
        px1_sum = torch.sum(px1,dim=0,keepdim=True)
        px1_delta = (px1_sum-px1).squeeze(-1).squeeze(-1)

        x = self.IA9(x, torch.cat([IA_head,px1_delta],dim=1))
        x = self.ASPP(x)

        x_emb_cur_1 = x.detach() 
        if memory_list[0]==None or x_emb_cur_1.size()!=memory_list[0].size():
            memory_list[0] = x_emb_cur_1
        x = self.Modulator_1(x, memory_list[0].cuda(x.device),IA_head)
        x_emb_cur_2 = x.detach()
        if memory_list[1]==None or x_emb_cur_2.size()!=memory_list[1].size():
            memory_list[1] = x_emb_cur_2
        x = self.Modulator_2(x, memory_list[1].cuda(x.device),IA_head)

        x = self.decoder_final(x, low_level_feat, IA_head)

        fg_logit = self.IA_logit(x, IA_head, self.IA_final_fg)
        bg_logit = self.IA_logit(x, IA_head, self.IA_final_bg)

        pred = self.augment_background_logit(fg_logit, bg_logit)
        memory_list =[x_emb_cur_1.cpu(),memory_list[1].cpu()]
        return pred,memory_list

    def IA_logit(self, x, IA_head, IA_final):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        IA_output = IA_final(IA_head)
        IA_weight = IA_output[:, :c]
        IA_bias = IA_output[:, -1]
        IA_weight = IA_weight.view(n, c, 1, 1)
        IA_bias = IA_bias.view(-1)
        logit = F.conv2d(x, weight=IA_weight, bias=IA_bias, groups=n).view(n, 1, h, w)
        return logit

    def decoder_final(self, x, low_level_feat, IA_head):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bicubic', align_corners=True)

        low_level_feat = self.GCT_sc(low_level_feat)
        low_level_feat = self.conv_sc(low_level_feat)
        low_level_feat = self.bn_sc(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = torch.cat([x, low_level_feat], dim=1)

        px1 = torch.nn.functional.avg_pool2d(x,kernel_size=(x.size()[-2],x.size()[-1]),padding = 0)
        px1_sum = torch.sum(px1,dim=0,keepdim=True)
        px1_delta = (px1_sum-px1).squeeze(-1).squeeze(-1)

        x = self.IA10(x, torch.cat([IA_head,px1_delta],dim=1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        px1 = torch.nn.functional.avg_pool2d(x,kernel_size=(x.size()[-2],x.size()[-1]),padding = 0)
        px1_sum = torch.sum(px1,dim=0,keepdim=True)
        px1_delta = (px1_sum-px1).squeeze(-1).squeeze(-1)

        x = self.IA11(x, torch.cat([IA_head,px1_delta],dim=1))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def Modulator_1(self, x, x_memory,IA_head):
        x = torch.cat([x, x_memory], dim=1)
        x = self.M1_Reweight_Layer_1(x, IA_head)
        x = self.M1_Bottleneck_1(x)
        x = self.M1_Reweight_Layer_2(x, IA_head)
        x = self.M1_Bottleneck_2(x)
        x = self.M1_Reweight_Layer_3(x, IA_head)
        x = self.M1_Bottleneck_3(x)
        return x

    def Modulator_2(self, x, x_memory,IA_head):
        x = torch.cat([x, x_memory], dim=1)
        x = self.M2_Reweight_Layer_1(x, IA_head)
        x = self.M2_Bottleneck_1(x)
        x = self.M2_Reweight_Layer_2(x, IA_head)
        x = self.M2_Bottleneck_2(x)
        x = self.M2_Reweight_Layer_3(x, IA_head)
        x = self.M2_Bottleneck_3(x)
        return x


    def augment_background_logit(self, fg_logit, bg_logit):
        #  Augment the logit of absolute background by using the relative background logit of all the 
        #  foreground objects.
        obj_num = fg_logit.size(0)
        pred = fg_logit
        if obj_num > 1:
            bg_logit = bg_logit[1:obj_num, :, :, :]
            aug_bg_logit, _ = torch.min(bg_logit, dim=0, keepdim=True)
            pad = torch.zeros(aug_bg_logit.size(), device=aug_bg_logit.device).expand(obj_num - 1, -1, -1, -1)
            aug_bg_logit = torch.cat([aug_bg_logit, pad], dim=0)
            pred = pred + aug_bg_logit
        pred = pred.permute(1,0,2,3)
        return pred


class DynamicPreHead(nn.Module):
    def __init__(self, in_dim=3, embed_dim=100, kernel_size=1):
        super(DynamicPreHead,self).__init__()
        self.conv=nn.Conv2d(in_dim,embed_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2))
        self.bn = nn.GroupNorm(int(embed_dim / 4), embed_dim)
        self.relu = nn.ReLU(True)
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
