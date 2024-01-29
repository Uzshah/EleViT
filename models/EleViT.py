import torch
import torch.nn as nn
import math
from timm.models.registry import register_model
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_



def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Input:
        in_channels: input channels
        out_channels: output channels
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs//2, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs//2),
        nn.ReLU(),
        nn.MaxPool2d(3, 2, padding = 1),
        nn.Conv2d(out_chs//2, out_chs, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), 
        nn.MaxPool2d(3, 2, padding = 1),)
    
    
class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, dilation_rate=1,
                 num_heads=8,bias=False, drop_out = 0.0):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        # Split the input into multiple heads
        self.qkv_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 3, kernel_size, 
                          1, padding, bias=bias, dilation=dilation_rate),
                nn.BatchNorm2d(out_channels * 3)
            )
        self.apply(self._init_weights)
        self.scale = math.sqrt(self.head_dim)
        self.soft = nn.Softmax(dim = -1)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.talking_head1 = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1, stride= 1, padding=0)
        #self.talking_head2 = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1, stride=1, padding=0)
        #self.gamma = nn.Parameter(torch.zeros(1))
        #self.attention_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.focusing_factor = nn.Parameter(torch.zeros(1))
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_out) if drop_out> 0 else nn.Identity()
        
        
  
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, x):
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (c h) w j -> (b h) c w j', h = self.num_heads),(q, k, v))
        scores = torch.einsum('b c w j,b c j k->b c w k', q, k.transpose(-1, -2))/self.scale
        # Add the parameterized attention bias term here
        #scores += self.attention_bias.view(1, self.head_dim, 1, 1)
        #scores = self.talking_head1(scores)
        attention_weights = self.soft(scores)
        #attention_weights = self.talking_head2(attention_weights)
        multihead_output = torch.einsum('b c w j,b c w j->b c w j', attention_weights*self.focusing_factor, v)
        out = rearrange(multihead_output, '(b h) c w j -> b (c h) w j', h = self.num_heads)
        out = self.drop(out)
        out = self.act(out)
        return self.bn(out)


class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_channels, out_channels=64, kernel_size=3, stride=1):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x
        

    
class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=3, stride=1, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(3, 2, padding = 1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.maxpool(x)
        x = self.norm(x)
        return x

    
class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttnFFN(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0.):
        super().__init__()
        self.conv = ConvEncoder(dim, dim)
        self.attn = MultiheadAttention(dim, dim, kernel_size = 3, padding=1, stride= 1, num_heads = 4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        
    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.attn(x)
        return x    
    
    
class EleViT(nn.Module):
    def __init__(self, num_classes=100, d_model=256, depth = 12, drop_path=0.):
        super(EleViT, self).__init__()
        
        self.drop_path = drop_path
        self.stem = stem(3, 40)
        
        self.attn1 =self._make_layer(40, 2)
        self.embed1 = Embedding(in_chans=40, embed_dim = 80)
        self.attn2 = self._make_layer(80, 2)
        self.embed2 = Embedding(in_chans=80, embed_dim = 256)
        self.attn3 = self._make_layer(256, 4)
        self.embed3 = Embedding(in_chans=256, embed_dim = 512)
        self.attn4 = self._make_layer(512, 2)
        
        ## classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dist_head = nn.Linear(
                    512, num_classes) if num_classes > 0 \
                    else nn.Identity()
        self.head = nn.Linear(
                512, num_classes) if num_classes > 0 \
                else nn.Identity()
    
    def _make_layer(self, dim, num):
        layers = []
        for _ in range(num):
            layers.append(AttnFFN(dim, drop_path = self.drop_path))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)   
        for layer in self.attn1:
            x = layer(x)
        x = self.embed1(x)
        for layer in self.attn2:
            x = layer(x)
        x = self.embed2(x)
        for layer in self.attn3:
            x = layer(x)
        x = self.embed3(x)
        for layer in self.attn4:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_out = (self.head(x)+self.dist_head(x))/2
        # For image classification
        return cls_out
    
@register_model
def EleViT_L(pretrained=False, num_classes=10, **kwargs):
    model = EleViT(num_classes)
    return model
