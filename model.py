
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from nn_util import get_act_layer, conv, unfoldNd

from Mynet.atten3D import VFRB ,cross_Sim
from networks import Unet, ConvBlock
import utils
def tuple_(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * length)




class Encoder(nn.Module):
    def __init__(self, in_c=1, c=4):
        super(Encoder, self).__init__()
      
        act=("leakyrelu", {"negative_slope": 0.1})  
        norm= 'instance'
        self.conv0 = double_conv(c, 2*c, act=act, norm=norm,
                        pre_fn=conv(in_c,c,act=act,norm=norm))
         
        self.conv1 =  double_conv(2 * c, 4 * c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))
  
        self.conv2 = double_conv(4 * c, 8 * c, act=act, norm=norm,
                        pre_fn= nn.AvgPool3d(2))

        self.conv3 = double_conv(8 * c, 16* c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))

        self.conv4 =double_conv(16 * c, 32 * c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))

    def forward(self, x):
        # out0 = self.conv0(x)  # 1
        out1 = self.conv1(x)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/16
        # print(out0.shape, out1.shape, out2.shape, out3.shape)
        return  out1, out2, out3, out4


def double_conv(in_c, out_c, act, norm='instance', append_fn=None, pre_fn=None):
    layer = nn.Sequential(pre_fn if pre_fn else nn.Identity(),
                          conv(in_c,  out_c, 3,1,1,act=act,norm=norm),
                          conv(out_c, out_c, 3,1,1,act=act,norm=norm),
                          append_fn if append_fn else nn.Identity()
                        )
    return layer

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class HVR_Net(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_c = 1,
                 ch_scale = 4,
                 num_k = 5, 
                 scale = 1.,
                 mean_type='s'
                ):
        super(HVR_Net, self).__init__()
        self.ch_scale = ch_scale
        self.inshape = inshape
        self.scale = scale
        c = self.ch_scale
        self.mt = mean_type
        if type(num_k) is not tuple:
            self.num_k = tuple_(num_k, length=4)
        else: self.num_k = num_k
        self.encoder = Encoder(in_c=8, c=c)
        act=("leakyrelu", {"negative_slope": 0.1}) 
        nb_feat_extractor = [[16] * 2, [8] * 4]
        
        self.reg_head = RegistrationHead(
            in_channels=2*2*c,
            out_channels=3,
            kernel_size=3,
        )
        
        self.feature_extractor = Unet(inshape,
                                    infeats=1,
                                    nb_features=nb_feat_extractor,
                                    nb_levels=None,
                                    feat_mult=1,
                                    nb_conv_per_level=1,
                                    half_res=False)        
        proj_n = 1
        self.up_tri = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv1 = double_conv(2*c, 2*c, act=act)
        self.cross_sim = cross_Sim()
        
        
        
        self.cbam00 = VFRB(in_channels=64*c,kernel_size=3)
        self.cbam0 = VFRB(in_channels=32*c,kernel_size=3)
        self.cbam1 = VFRB(in_channels=16*c,kernel_size=3)
        self.cbam2 = VFRB(in_channels=8*c,kernel_size=3)
        self.cbam3 = VFRB(in_channels=4*c,kernel_size=3)
        
        
        self.conv1_out = double_conv(2*2*c, 2*c, act=act, append_fn=conv(2*c,3, 3,1,1, act=None))
        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(utils.SpatialTransformer([s // 2**i for s in inshape]))

    def set_k(self, k):
        
        if type(k) is not tuple:
            k = tuple_(k, length=4)
        # self.sacb_proj5.set_num_k(k[0])
        # self.sacb_proj4.set_num_k(k[1])
        # self.sacb_proj3.set_num_k(k[2])
        # self.sacb_proj2.set_num_k(k[3])
        
    def forward(self, x, y, softsign_last=False):


        M1 = self.feature_extractor(x)
        F1 = self.feature_extractor(y)
        # encode stage
        M2, M3, M4, M5 = self.encoder(M1)
        F2, F3, F4, F5 = self.encoder(F1)


        M5 = self.cbam00(  torch.cat([M5, F5],dim=1) )
        M5 ,F5 = M5.split([32*self.ch_scale, 32*self.ch_scale], dim=1)
        
        phi_5 = self.cross_sim(M5, F5)
        phi_5 = self.up_tri(2.* phi_5)

        # F4 = self.sdm0(F4, M4)

        M4 = self.transformer[3](M4, phi_5)
        M4 = self.cbam0(torch.cat([M4, F4],dim=1) )
        M4 ,F4 = M4.split([16*self.ch_scale, 16*self.ch_scale], dim=1)
        
        phi_4 = self.cross_sim(M4, F4) 
        phi_4 = self.up_tri(2.* (self.transformer[3](phi_5, phi_4) + phi_4))

        M3 = self.transformer[2](M3, phi_4)

        # M3 = self.sdm1(M3, F3)
        F3 = self.cbam1(torch.cat([M3, F3],dim=1) )
        M3 ,F3 = F3.split([8*self.ch_scale, 8*self.ch_scale], dim=1)
        
        phi_3 = self.cross_sim(M3, F3)
        phi_3 = self.up_tri(2.* (self.transformer[2](phi_4, phi_3) + phi_3))

        M2 = self.transformer[1](M2, phi_3)
        # F2 = self.sdm2(M2, F2)
        M2 = self.cbam2(torch.cat([M2, F2],dim=1) )
        M2 ,F2 = M2.split([4*self.ch_scale, 4*self.ch_scale], dim=1)
        phi_2 = self.cross_sim(M2, F2)
        phi_2 = self.up_tri(2.* (self.transformer[1](phi_3, phi_2) + phi_2))
            


        M1 = self.transformer[0](M1, phi_2)
        
        F1, M1 = self.conv1(F1), self.conv1(M1)
        M1 = self.cbam3(torch.cat([M1, F1],dim=1) )
        delta_phi_1 = self.reg_head(M1)
        if softsign_last:
            delta_phi_1 = F.softsign(delta_phi_1)
        # w = self.conv1_out(torch.cat([M1,F1],1))
        Phi = self.transformer[0](phi_2, delta_phi_1) + delta_phi_1
        
        x_warped = self.transformer[0](x, Phi)
        return x_warped, Phi
        