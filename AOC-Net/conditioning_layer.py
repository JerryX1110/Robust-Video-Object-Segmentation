import torch
import torch.nn as nn
import torch.nn.functional as F


class conditioning_layer(nn.Module):
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
