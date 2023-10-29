import torch
import torch.nn as nn
import math
from timm.models.registry import register_model
from einops import rearrange

def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Input:
        in_channels: input channels
        out_channels: output channels
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs //2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs//2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation_rate=1, bias=False, num_heads = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        # Split the input into multiple heads
        self.query_heads = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                                 padding, bias=bias, dilation=dilation_rate),
                                                        nn.BatchNorm2d(out_channels))
        self.key_heads = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                                 padding, bias=bias, dilation=dilation_rate),
                                                        nn.BatchNorm2d(out_channels))
        self.value_heads = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                                 padding, bias=bias, dilation=dilation_rate),
                                                        nn.BatchNorm2d(out_channels))

        # Modify the residual connection
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False, dilation=dilation_rate,padding=1),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        query = self.query_heads(x)
        key = self.key_heads(x)
        value = self.value_heads(x)

        # Calculate scaling factor

        scores = torch.einsum('b c w j,b c j k->b c w k', query, key.transpose(-1, -2))
        attention_weights = nn.functional.softmax(scores, dim=-1)

        head_output = torch.einsum('bchw,bchw->bchw', attention_weights, value)  # Attention output for this head

        # Residual connection and further processing
        residual = self.downsample(x)
        out = head_output + residual
        return out
   

class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation_rate=1,
                 num_heads=8,bias=False):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        # Split the input into multiple heads
        self.query = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                                    padding, bias=bias, dilation=dilation_rate),
                                                    nn.BatchNorm2d(out_channels))
        self.key = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                                    padding, bias=bias, dilation=dilation_rate),
                                                    nn.BatchNorm2d(out_channels))
        self.value = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                                    padding, bias=bias, dilation=dilation_rate),
                                                    nn.BatchNorm2d(out_channels))

        self.scale = math.sqrt(self.head_dim)
        self.soft = nn.Softmax(dim = -1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.focusing_factor = nn.Parameter(torch.zeros(1))
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False, dilation=dilation_rate),
                nn.BatchNorm2d(out_channels)
            ) if (stride != 1 or in_channels != out_channels) else nn.Identity()
            

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        q = rearrange(query, 'b (c h) w j -> b c h w j', h = self.num_heads)
        k = rearrange(key, 'b (c h) w j -> b c h w j', h = self.num_heads)
        v = rearrange(value, 'b (c h) w j -> b c h w j', h = self.num_heads)
        scores = torch.einsum('b c h w j,b c h j k->b c h w k', q, k.transpose(-1, -2))/self.scale
        attention_weights = self.soft(scores)
        multihead_output = torch.einsum('b c h w j,b c h w j->b c h w j', attention_weights, v*self.focusing_factor)
        multihead_output = rearrange(multihead_output, 'b c h w j -> b (c h) w j')
        residual = self.downsample(x)
        out = self.gamma * multihead_output + residual
        out = self.bn(out)
        return out


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
        # Modify the residual connection
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, bias=False, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        

    def forward(self, x):
        input = x
        #print(x.size())
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        #print(x.size())
        res = self.downsample(input)
        #print(res.size())
        x = res + x
        return x
        

    
class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=3, stride=2, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
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
    
class EleViT(nn.Module):
    def __init__(self, num_classes=100, d_model=256, depth = 12):
        super(EleViT, self).__init__()
        
        self.stem = stem(3, 128)
        
        self.attn1 = nn.ModuleList([nn.Sequential(ConvEncoder(128, 128),
                                                  MultiheadAttention(128, 128)) for _ in range(2)])
        self.embed1 = Embedding(in_chans=128, embed_dim = 256)
        
        self.attn2 = nn.ModuleList([nn.Sequential(ConvEncoder(d_model, d_model),
                                                  MultiheadAttention(256, 256)) for _ in range(3)])
        self.embed2 = Embedding(in_chans=256, embed_dim = 288)
        
        self.attn3 = nn.ModuleList([nn.Sequential(ConvEncoder(288, 288),
                                                  MultiheadAttention(288,288)) for _ in range(3)])
        self.embed3 = Embedding(in_chans=288, embed_dim = 384)
        
        self.attn4 = nn.ModuleList([nn.Sequential(ConvEncoder(384,384),
                                                  MultiheadAttention(384,384)) for _ in range(2)])
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dist_head = nn.Linear(
                    384, num_classes) if num_classes > 0 \
                    else nn.Identity()
  
        self.head = nn.Linear(
                384, num_classes) if num_classes > 0 \
                else nn.Identity()
        
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