import torch
import torch.nn as nn
import math
from timm.models.registry import register_model


def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Input:
        in_channels: input channels
        out_channels: output channels
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), 
        nn.Conv2d(out_chs, out_chs *2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs* 2),
        nn.ReLU(),
        nn.Conv2d(out_chs * 2, out_chs, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(),
    )

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation_rate=1, bias=False, num_heads = 4):
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
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, bias=False, dilation=dilation_rate,padding=1),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        query = self.query_heads(x)
        key = self.key_heads(x)
        value = self.value_heads(x)

        # Calculate scaling factor

        scores = torch.einsum('bchw,bchw->bchw', query, key.transpose(-1, -2))
        attention_weights = nn.functional.softmax(scores, dim=-1)

        head_output = torch.einsum('bchw,bchw->bchw', attention_weights, value)  # Attention output for this head

        # Residual connection and further processing
        residual = self.downsample(x)
        out = head_output + residual
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

    def __init__(self, patch_size=3, stride=1, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=2, padding=padding)
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
    
class MVT(nn.Module):
    def __init__(self, num_classes=200):
        super(MVT, self).__init__()
        
        self.stem = stem(3, 256)
        self.stage1 = nn.Sequential(ConvEncoder(256, 256), Attention(256, 256), Embedding(in_chans=256, embed_dim = 256))
        self.stage2 = nn.Sequential(ConvEncoder(256, 256), Attention(256, 256), Embedding(in_chans=256, embed_dim = 256))
        self.stage3 = nn.Sequential(ConvEncoder(256, 256), Attention(256, 256),
                                    ConvEncoder(256, 256), Attention(256, 256), 
                                    ConvEncoder(256, 256), Attention(256, 256), ConvEncoder(256, 256), Attention(256, 256))
        self.stage4 = nn.Sequential(ConvEncoder(256, 256), Attention(256, 256), ConvEncoder(256, 256), Attention(256, 256), 
                                    ConvEncoder(256, 256), Attention(256, 256))
        self.stage5 = nn.Sequential(ConvEncoder(256, 256), Attention(256, 256), ConvEncoder(256, 256), Attention(256, 256), 
                                    ConvEncoder(256, 256), Attention(256, 256))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(
                256, num_classes) if num_classes > 0 \
                else nn.Identity()
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        # For image classification
        return cls_out
    
@register_model
def MVT_L3(pretrained=False, **kwargs):
    model = MVT(200)
    return model