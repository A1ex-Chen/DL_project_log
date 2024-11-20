# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch.nn as nn

from src.models.model_utils import SqueezeAndExcitation ,scSE,ChSqueezeAndSpExcitation,SpSqueezeAndChExcitation,ESqueezeAndExcitation,SPALayer,SPABLayer,SPACLayer,CBAM,BAM,SpatialGroupEnhance


class SqueezeAndExciteFusionAdd(nn.Module):
        
    #    #*TripletAttention
    #     self.se_rgb = TripletAttention(channels_in)
    #     self.se_depth =TripletAttention(channels_in)
        
        
          # * CBAM
        # self.se_rgb = CBAM(channels_in)
        # self.se_depth = CBAM(channels_in)
        
        #*BAM
        # self.se_rgb = BAM(channels_in)
        # self.se_depth =BAM(channels_in)
        
        # *sSE 
        # self.se_rgb = ChSqueezeAndSpExcitation(channels_in)
        # self.se_depth = ChSqueezeAndSpExcitation(channels_in)
        
    #    # *scSE
        # self.se_rgb = scSE(channels_in)
        # self.se_depth = scSE(channels_in)
        
        # *SPALayer
        # self.se_rgb = SPALayer(channels_in,
        #                         activation=activation)
        # self.se_depth =SPALayer(channels_in,
        #                         activation=activation) 
        # *SPABLayer
        # self.se_rgb = SPABLayer(channels_in,
        #                         activation=activation)
        # self.se_depth =SPABLayer(channels_in,
        #                         activation=activation) 
        #   # *SPACLayer
        # self.se_rgb = SPACLayer(channels_in,
        #                         activation=activation)
        # self.se_depth =SPACLayer(channels_in,
        #                         activation=activation) 
        
        # # *cSE
        # self.se_rgb = SpSqueezeAndChExcitation(channels_in)
        # self.se_depth = SpSqueezeAndChExcitation(channels_in)
            

    
class ESqueezeAndExciteFusionAdd(nn.Module):
