
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from ..core import register

__all__ = ['CFDFEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)


        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()
        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch, act='silu'):

        super(HWD, self).__init__()

        try:
            from pytorch_wavelets import DWTForward
            self.wt = DWTForward(J=1, mode='zero', wave='haar')
        except ImportError:
            raise ImportError("pytorch_wavelets is not installed. Please install it using: pip install pytorch-wavelets")

        self.conv = ConvNormLayer_fuse(in_ch * 4, out_ch, 1, 1, padding=0, act=act)

    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, x):

        yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

        x = self.conv(x)
        return x

class FWD(nn.Module):
    def __init__(self, c1, c2, act='mish'):  # ch_in, ch_out

        super().__init__()

        assert c1 % 2 == 0, f"输入通道数c1={c1}必须能被2整除"

        assert c2 % 2 == 0, f"输出通道数c2={c2}必须能被2整除"

        self.c = c2 // 2
        self.cv1 = HWD(c1 // 2, self.c, act=act)
        self.cv2 = ConvNormLayer_fuse(c1 // 2, self.c, 1, 1, padding=0, act=act)

    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, x):
        # （stride=1, padding=0，kernel_size=2）
        x = F.avg_pool2d(x, 2, 1, 0, False, True)

        x1, x2 = x.chunk(2, 1)

        x1 = self.cv1(x1)

        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)

        return torch.cat((x1, x2), 1)

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', **kwargs):

        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer_fuse):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

# ========================================

def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    elif act == 'silu':
        layer = nn.SiLU(inplace)
    elif act == 'mish':
        layer = nn.Mish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def gcd(a, b):

    while b:
        a, b = b, a % b
    return a

class MSDC(nn.Module):

    def __init__(self, in_channels, kernel_sizes, stride, mscb_act='mish', dw_parallel=True):  # 将act改为mscb_act
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.mscb_act = mscb_act  # 改为mscb_act
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.mscb_act, inplace=True)  # 改为mscb_act
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):

    def __init__(self, ch_in, ch_out, act='mish', shortcut=True, stride=1,
                 kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, mscb_act=None):
        super(MSCB, self).__init__()

        if mscb_act is not None:
            self.mscb_act = mscb_act
        else:
            self.mscb_act = act

        self.in_channels = ch_in
        self.out_channels = ch_out
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = shortcut
        self.n_scales = len(self.kernel_sizes)

        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.mscb_act, inplace=True)  # 使用mscb_act
        )

        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride,
                         self.mscb_act, dw_parallel=self.dw_parallel)  # 传递mscb_act

        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)

        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)

        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock,
                 **kwargs):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act, **kwargs) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu",
                 csp_type='csp2',
                 bottletype=VGGBlock,
                 **kwargs):  # 添加**kwargs参数
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)

        if csp_type == 'csp2':
            SelectedCSPLayer = CSPLayer2
        else:
            SelectedCSPLayer = CSPLayer

        self.cv2 = nn.Sequential(
            SelectedCSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act,
                           bottletype=bottletype, **kwargs),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv3 = nn.Sequential(
            SelectedCSPLayer(c4, c4, n, 1, bias=bias, act=act,
                           bottletype=bottletype, **kwargs),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# This layer is equivalent to RepC3 in YOLOs repo
class CSPLayer2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock,
                 **kwargs):
        super(CSPLayer2, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels * 2, 1, 1, bias=bias, act=act)
        # 传递kwargs给bottleneck
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act, **kwargs) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))


class RepNCSPELAN5(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu",
                 bottletype=VGGBlock, **kwargs):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)

        # 传递kwargs
        self.cv2 = nn.Sequential(
            CSPLayer2(c3 // 2, c4, n, 1, bias=bias, act=act,
                      bottletype=bottletype, **kwargs)
        )
        self.cv3 = nn.Sequential(
            CSPLayer2(c4, c4, n, 1, bias=bias, act=act,
                      bottletype=bottletype, **kwargs)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        out = self.cv4(torch.cat(y, 1))
        return out

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        out = self.cv4(torch.cat(y, 1))
        return out


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="gelu",
                 normalize_before=False,
                 ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class CFDFEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='mish',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='mish',
                 eval_spatial_size=None,
                 version='wdd',
                 csp_type='csp2',
                 fuse_op='cum',
                 downsample_type='fwd',

                 use_mscb=True,
                 mscb_kernel_sizes=[1, 3, 5, 7],
                 mscb_expansion_factor=2,
                 mscb_dw_parallel=True,
                 mscb_activation='mish',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op
        self.downsample_type = downsample_type

        self.use_mscb = use_mscb
        self.mscb_kernel_sizes = mscb_kernel_sizes
        self.mscb_expansion_factor = mscb_expansion_factor
        self.mscb_dw_parallel = mscb_dw_parallel
        self.mscb_activation = mscb_activation

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2  # deim use sum instead of cat

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)


        if  downsample_type == 'fwd':

            fwd_in_ch = hidden_dim  #hidden_dim 或 input_dim
            if fwd_in_ch % 2 != 0:
                fwd_in_ch += 1
            SCDown_Conv = FWD(fwd_in_ch, hidden_dim, act=act)
        else:

            SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim * 2, round(
            expansion * hidden_dim // 2), round(3 * depth_mult)

        if self.use_mscb:

            BottleneckType = MSCB

            bottletype_kwargs = {
                'kernel_sizes': mscb_kernel_sizes,
                'expansion_factor': mscb_expansion_factor,
                'dw_parallel': mscb_dw_parallel,
                'mscb_act': mscb_activation,
            }
        else:

            BottleneckType = VGGBlock
            bottletype_kwargs = {}

        if version == 'dfine':
            Fuse_Block = RepNCSPELAN4(
                c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks,
                act=act, csp_type=csp_type,
                bottletype=BottleneckType,
                **bottletype_kwargs
            )
        elif version == 'wdd':
            Fuse_Block = RepNCSPELAN5(
                c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks,
                act=act,
                bottletype=BottleneckType,
                **bottletype_kwargs
            )
        else:
            Fuse_Block = CSPLayer(
                in_channels=c1, out_channels=c2,
                num_blocks=num_blocks, act=act,
                expansion=expansion,
                bottletype=BottleneckType,
                **bottletype_kwargs
            )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='bilinear')   #`nearest`, `linear` (3D-only),`bilinear`, `bicubic` (4D-only). Default: 'nearest'
            fused_feat = (upsample_feat + feat_low) \
                if self.fuse_op == 'sum' else torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](fused_feat)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]

            if self.downsample_type in ['fwd']:
                if feat_low.size(1) % 2 != 0:
                    feat_low = F.pad(feat_low, (0, 0, 0, 0, 0, 1), "constant", 0)

            downsample_feat = self.downsample_convs[idx](feat_low)
            fused_feat = (downsample_feat + feat_height) \
                if self.fuse_op == 'sum' else torch.concat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs