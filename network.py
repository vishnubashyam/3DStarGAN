from munch import Munch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import copy

class ResBlk(nn.Module):
    def __init__(self, dim_in, 
                dim_out, actv=nn.LeakyReLU(0.2),
                normalize=False, downsample=False):
        """Residual Block
        Args:
            dim_in (int): Input channels
            dim_out (int): Output channels
            actv (nn.Module, optional): Activation function. Defaults to nn.LeakyReLU(0.2).
            normalize (bool, optional): Normalize using instance norm. Defaults to False.
            downsample (bool, optional): Downsample via average polling 2x2x2 kernel. Defaults to False.
        """
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv3d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm3d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm3d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1x1 = nn.Conv3d(dim_in, dim_out, 1, 1, 0, bias=False)
    
    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1x1(x)
        if self.downsample:
            x = F.avg_pool3d(x, 2)
        return x
    
    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool3d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    
    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        """Adaptive Instance Normalization

        Args:
            style_dim (int): Size of the style encoding
            num_features (int): Number of channels in input
        """
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                    actv=nn.LeakyReLU(0.2), upsample=False):
        """Resnet block with adaptive instance normalization

        Args:
            dim_in (int): Input channels
            dim_out (int): Output channels
            style_dim (int, optional): Size of the style encoding. Defaults to 64.
            actv (nn.Module, optional): Defaults to nn.LeakyReLU(0.2).
            upsample (bool, optional): Defaults to False.
        """
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv3d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1x1 = nn.Conv3d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2
        self.img_size = img_size
        self.from_rgb = nn.Conv3d(1, dim_in, 3, 1, 1) 
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm3d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim_in, 1, 1, 1, 0))## Padding added to fix output size - need to figure out why output img size is wrong

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        identity_connection = []
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            identity_connection.insert(0, x)
        i=0
        for block, identity in zip(self.decode, identity_connection):
            if i>2:
                x = torch.add(x, identity)
            i+=1
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 4
        blocks = []
        blocks += [nn.Conv3d(1, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv3d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(256, style_dim)] ### 864 was hardcoded, originally dim_out, varies as img size varies
    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            print(h.size())
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y.long()]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**10 // img_size
        blocks = []
        blocks += [nn.Conv3d(1, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv3d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv3d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)

        out = out[idx, (y.long())]  # (batch)
        return out


def build_model(args):
    generator = Generator(args.img_size, args.style_dim)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                mapping_network=mapping_network,
                style_encoder=style_encoder,
                discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                mapping_network=mapping_network_ema,
                style_encoder=style_encoder_ema)

    return nets, nets_ema
