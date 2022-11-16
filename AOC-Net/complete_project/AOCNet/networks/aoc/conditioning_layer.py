import torch
import torch.nn as nn
import torch.nn.functional as F


class conditioning_layer(nn.Module):

    # Equation (7) of the main paper
    
    def __init__(self,
            in_dim=256,
            beta_percentage=0.3):
        super(conditioning_layer,self).__init__()

        self.beta_percentage = beta_percentage 

        kernel_size = 1
        self.phi_layer = nn.Conv2d(in_dim,1,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2))    
        self.mlp_layer = nn.Linear(in_dim, in_dim)

        nn.init.kaiming_normal_(self.phi_layer.weight,mode='fan_out',nonlinearity='relu')


    def forward(self, z_in):

        # Step 1: phi(z_in)
        x = self.phi_layer(z_in)  

        # Step 2: beta
        x = x.reshape(x.size()[0],x.size()[1],-1)
        z_in_reshape = z_in.reshape(z_in.size()[0],z_in.size()[1],-1)
        beta_rank = int(self.beta_percentage*z_in.size()[-1]*z_in.size()[-2])
        beta_val, _ = torch.topk(x, k=beta_rank, dim=-1, sorted=True)
        
        # Step 3: pi_beta(phi(z_in))
        x = x > beta_val[...,-1,None]

        # Step 4: z_in \odot pi_beta(phi(z_in))
        z_in_masked = z_in_reshape * x

        # Step 5: GAP (z_in \odot pi_beta(phi(z_in)))
        z_in_masked_gap = torch.nn.functional.avg_pool1d(z_in_masked, 
                                                              kernel_size=z_in_masked.size()[-1]).squeeze(-1)

        # Step 6: MLP ( GAP (z_in \odot pi_beta(phi(z_in))) )
        out = mlp_layer(z_in_masked_gap)
        
        return out

class conditioning_block(nn.Module):
    
    # Equation (5) of the main paper

    def __init__(self,
            in_dim=256,
            proxy_dim = 400,
            beta_percentage=0.3):
        super(conditioning_block,self).__init__()

        self.CL_1 = conditioning_layer(in_dim, beta_percentage)
        self.CL_2 = conditioning_layer(in_dim, beta_percentage)
        self.CL_3 = conditioning_layer(proxy_dim, 1) 

        self.mlp_layer = nn.Linear(in_dim * 2 + proxy_dim, in_dim)

    def forward(self, x, proxy_IA_head):
        
        px1 = torch.nn.functional.avg_pool2d(x,kernel_size=(x.size()[-2],x.size()[-1]),padding = 0)
        x_delta = (torch.sum(px1,dim=0,keepdim=True)-px1).squeeze(-1).squeeze(-1)

        # Step 1: cal intra-object conditioning code
        cl_out_1 = CL_1(x)

        # Step 2: cal inter-object conditioning code
        cl_out_2 = CL_2(x_delta)

        # Step 3: cal conditioning code with poxies
        cl_out_3 = CL_3(proxy_IA_head)

        # Step 4: conduct calibration
        a = self.mlp_layer(torch.cat([cl_out_1,cl_out_2,cl_out_3],dim=1))
        a = 1. + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        
        return x
