#An MLP-Mixer (Multi-Layer Perceptron Mixer) is a neural network architecture introduced as an alternative to convolutional neural networks (CNNs) and transformers. It relies solely on multi-layer perceptrons (MLPs) to process both spatial and channel-wise information in an image, making it simpler than transformer-based models.

import math
from functools import partial
import torch
import torch.nn as nn #used for defining the nerual networks

#providing specialized layers like mlp
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple

#math operations
import collections.abc
import math
import re
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, Iterator, Tuple, Type, Union

#enables memory efficient training
import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint


#Recursively applies a function (fn) to all layers of a PyTorch model (module).
#Initializing weights or modifying model behavior.
    #Works in a DFS manner 
def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
        ) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


#CORE COMPUTATIONAL BLOCK!!!
class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            hmr_embedd_dim,
            seq_len=145, #Previously number patches 
            mlp_ratio=(0.5, 4.0), #Only determines hidden states 
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        #computer token and channel hidden dimensions
        tokens_dim, channels_dim = [int(x * hmr_embedd_dim) for x in to_2tuple(mlp_ratio)] #Tokens_dim and channels_dim are both hidden states (256, 2048)
        #defining layers
        self.tokens_dim = tokens_dim
        self.channels_dim = channels_dim
        self.norm1 = norm_layer(hmr_embedd_dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop) #145 => 256 => 145   
        self.mlp_channels = mlp_layer(hmr_embedd_dim, channels_dim, act_layer=act_layer, drop=drop) #512 => 2048 => 512 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(hmr_embedd_dim)

    def forward(self, x): 
        #A forward pass means passing the input through the layers of the neural network to compute an output.
        '''token mixing: 
            Normalize x.
            Transpose it to switch patches and features.
            Apply mlp_tokens.
            Transpose back.
            Add result to x.'''
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        
        '''channel mixing:
            Normalize x.
            Apply mlp_channels.
            Add result to x.'''
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


#Used to modulate activations after normalization.
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)
    

#The main MLP-Mixer model.
#Contains multiple MixerBlocks stacked together.

class MlpMixer(nn.Module):
    #It consists of two main MLP layers:
    #Token Mixing MLP - Mixes information across spatial dimensions (i.e., patches of an image).
    #Channel Mixing MLP - Mixes information across feature channels (i.e., per patch).
    
    def __init__(
            self,
            num_classes=3,
            num_blocks=8,
            hmr_embedd_dim=512, #number of features per input token
            seq_len=145, #number of tokens
            mlp_ratio=(0.5, 4.0), #size of hidden layers
            block_layer=MixerBlock,
            mlp_layer=Mlp, #type of MLP
            norm_layer=partial(nn.LayerNorm, eps=1e-6), #normalization function
            act_layer=nn.GELU, #activation function
            #dropout rates
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            global_pool='avg',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = hmr_embedd_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.blocks = nn.Sequential(*[

            #Creates num_blocks (default 8) Mixer Blocks.
            #Each block processes the input with MLP layers and normalization.
            block_layer(
                hmr_embedd_dim,
                seq_len,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(num_blocks)])
        
        #I DONT UNDERSTAND THESE CONCEPTS!!!!!!!!!!!!! REFER LATER
        self.norm = norm_layer(hmr_embedd_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(hmr_embedd_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

#This function initializes weights for different layers (Linear, Conv, LayerNorm).

def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

if __name__ == "__main__":
    print("Starting") 
    model = MlpMixer()
    input = torch.rand((1, 145, 512))
    out = model(input)
    breakpoint() 