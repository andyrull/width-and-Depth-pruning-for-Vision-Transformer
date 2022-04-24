import math
from itertools import repeat
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from pruner.modules.masked_linear import MaskedLinear

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 fc_pruning=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = MaskedLinear(
            in_features,
            hidden_features,
            bias_mask=True,
            fc_pruning=fc_pruning,
        )
        self.act = act_layer()
        self.fc2 = MaskedLinear(
            hidden_features,
            out_features,  # config.intermediate_size,
            bias_mask=True,
            fc_pruning=fc_pruning
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 head_pruning=True,fc_pruning=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.att_output_mask = None
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = MaskedLinear(
            dim,
            dim * 3,
            head_split=num_heads,
            bias_mask=qkv_bias,
            head_pruning=head_pruning
        )
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = MaskedLinear(
            dim,
            dim,
            bias_mask=qkv_bias,#True,
            fc_pruning=fc_pruning
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # att_map = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def make_structural_pruning(self):
        # make the structural pruning here for attention for inference
        value_mask = self.qkv.make_inference_pruning()
        num_head = (value_mask.sum() // 64).item()
        self.num_heads = int(num_head)

        att_output_mask = self.proj.make_inference_pruning()  # we need this to do the residual connection
        self.att_output_mask = att_output_mask
        # we need to remove cols from O layer since heads are pruned
        self.proj.make_column_purning(value_mask)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 head_pruning=True,fc_pruning=True,block_pruning=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop,head_pruning=head_pruning,fc_pruning=fc_pruning)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       fc_pruning=fc_pruning)
        self.mlp_output_mask = None
        self.block_pruning=block_pruning

    def forward(self, x):
        if not self.block_pruning:
            return self.normal_forward(x)
        else:
            return self.block_pruning_forward(x)

    def normal_forward(self,x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)

        if self.attn.att_output_mask is not None:
            h[:, :, self.attn.att_output_mask] = h[:, :, self.attn.att_output_mask] + self.drop_path(x)
            x = h
        else:
            x = h + self.drop_path(x)

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.mlp_output_mask is not None:
            h[:, :, self.mlp_output_mask] = h[:, :, self.mlp_output_mask] + self.drop_path(x)
            x = h
        else:
            x = h + self.drop_path(x)
        return x

    def make_structural_pruning(self):
        self.attn.make_structural_pruning()
        # make self.intermediate and self.output to be structural prune
        intermediate_mask = self.mlp.fc1.make_inference_pruning()  # we need this to do the residual connection
        output_mask = self.mlp.fc2.make_inference_pruning()
        self.mlp_output_mask = output_mask
        # we need to do col pruning for FC2
        self.mlp.fc2.make_column_purning(intermediate_mask)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x


class VisionTransformerWithWDPruning(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 head_pruning=True,fc_pruning=True,classifiers=[3,7],
                 classifier_choose=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)
        self.classifiers = classifiers
        self.classifier_choose = classifier_choose
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))


        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if not self.classifier_choose:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    head_pruning=head_pruning,fc_pruning=fc_pruning)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    head_pruning=head_pruning, fc_pruning=fc_pruning)
                for i in range(self.classifier_choose)])


        self.norm = norm_layer(embed_dim)
        self.norm_classifier = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(classifiers))])

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_classifier = nn.ModuleList([nn.Linear(self.num_features, num_classes) for _ in range(len(classifiers))])

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_structural_pruning(self):
        for layer in range(len(self.blocks)):
            self.blocks[layer].make_structural_pruning()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        embed = []
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if (i+1) in self.classifiers:
                embed.append(x)

        x = self.norm(x)
        for i,norm in enumerate(self.norm_classifier):
            embed[i] = norm(embed[i])

        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), [self.pre_logits(y[:, 0]) for y in embed]
        else:
            return x[:, 0],[self.pre_logits(y[:, 0]) for y in embed], x[:, 1]

    def forward(self, x):
        if not self.classifier_choose:
            return self.train_forward(x)
        else:
            return self.inference_forward(x)

    def train_forward(self,x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            ret = []
            y, x_dist = self.head(x[0]), self.head_dist(x[2])  # x must be a tuple
            for i, hd in enumerate(self.head_classifier):
                ret.append(hd(x[1][i]))
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return y, ret,x_dist
            else:
                return (y + x_dist) / 2,ret
        else:
            ret = []
            y = self.head(x[0])
            for i,hd in enumerate(self.head_classifier):
                ret.append(hd(x[1][i]))
        return y,ret

    def inference_forward(self,x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x[:,0])

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

    def LayerPruningAndLoadParams(self,dir):

        param_dict = torch.load(dir)['model']
        if self.classifier_choose != 12:
            for i in param_dict:
                if 'block' in i:
                    block_i = int(i.replace('module.', '').split('.')[1])
                    if block_i < self.classifier_choose:
                        self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                elif 'head' in i:
                    idx = ''.join(list(filter(str.isdigit, i)))
                    if idx and int(idx) == self.classifiers.index(self.classifier_choose):
                        self.state_dict()[i.replace('module.', '').replace('_classifier.' + idx, '')].copy_(param_dict[i])
                elif 'norm' in i:
                    idx = ''.join(list(filter(str.isdigit, i)))
                    if idx and int(idx) == self.classifiers.index(self.classifier_choose):
                        self.state_dict()[i.replace('module.', '').replace('_classifier.' + idx, '')].copy_(param_dict[i])
                else:
                    self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        else:
            for i in param_dict:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        delattr(self, 'head_classifier')
        delattr(self, 'norm_classifier')

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict