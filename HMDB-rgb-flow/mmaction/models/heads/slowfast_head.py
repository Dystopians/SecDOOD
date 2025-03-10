'''
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import tenseal as ts
from ..registry import HEADS
from .base import BaseHead
from ..hypernetwork.hypernetwork_modules import HyperNetwork
from ..hypernetwork.hypernetwork_modules import HyperNetwork_Linear, HyperNetwork_Li, Embedding
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
def zero_topN_channels_by_info(X: torch.Tensor, N: int, bins: int = 256, mode='Info') -> torch.Tensor:
    assert X.ndim == 5
    if mode=='Info':
        B, T, H, W, C = X.shape
        assert N <= C
        X_cpu = X.cpu()
        entropies = []
        variances = []
        for c in range(C):
            channel_data = X_cpu[..., c].view(-1)
            var_c = channel_data.var().item()
            data_min = channel_data.min().item()
            data_max = channel_data.max().item()
            if data_min == data_max:
                ent_c = 0.0
            else:
                hist = torch.histc(channel_data, bins=bins, min=data_min, max=data_max)
                p = hist / hist.sum()
                p_nonzero = p[p > 0]
                ent_c = -(p_nonzero * p_nonzero.log()).sum().item()
            entropies.append(ent_c)
            variances.append(var_c)
        entropies = torch.tensor(entropies, dtype=torch.float32)
        variances = torch.tensor(variances, dtype=torch.float32)
        entropies_norm = torch.softmax(entropies, dim=0)
        variances_norm = torch.softmax(variances, dim=0)
        info_scores = 0.5 * entropies_norm + 0.5 * variances_norm
        scores_sorted, indices_sorted = torch.sort(info_scores, descending=True)
        # 反向遮掩是N:
        topN_indices = indices_sorted[N:]
        with torch.no_grad():
            for c_idx in topN_indices:
                X[..., c_idx] = 0.0
    elif mode == 'RandCh':
        B, T, H, W, C = X.shape
        num_to_zero = N
        random_indices = torch.randperm(C)[:num_to_zero]
        with torch.no_grad():
            for c_idx in random_indices:
                X[..., c_idx] = 0.0
    elif mode == 'Rand Half':
        numel = X.numel()
        B, T, H, W, C = X.shape
        num_to_zero = numel // C * N
        X_flat = X.view(-1)
        indices = torch.randperm(numel)[:num_to_zero]
        with torch.no_grad():
            X_flat[indices] = 0.0
    else:
        X = X

    return X
@HEADS.register_module()
class SlowFastHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 reduce_channel=False,
                 reduce_channel_num=2048,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.reduce_channel = reduce_channel
        self.reduce_fast = nn.Linear(reduce_channel_num, reduce_channel_num)
        self.reduce_channel_num = reduce_channel_num
        self.Hyper = False # Use adaptive cnn or not
        self.PCA = False # Training false/ Test True
        self.Enc = False and self.PCA # Keys
        self.Test = False # Training False Test True
        if self.Hyper:
            self.Hyper_slow = HyperNetwork(f_size=1, z_dim=256, out_size=256, in_size=256)
            self.Hyper_fast = HyperNetwork(f_size=1, z_dim=512, out_size=512, in_size=512)
            self.Hyper_Cls = HyperNetwork_Li(hidden=32, z_dim=768, label_size=25, batch_size=16, zoom_dim=768)
        if self.Enc:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, [60, 40, 40, 60])
            self.context.global_scale = 2 ** 40
            self.context.generate_galois_keys()
        if self.PCA:
            self.mode='None'
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        # if self.reduce_channel:
        #     print("reduce_channel")
        #     self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)
        #     self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)
        # else:
        #     self.fc_cls = nn.Linear(in_channels, num_classes)

        self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)

        # self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)
        self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        x_fast, x_slow = x
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace(   )
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        x_fast = self.avg_pool(x_fast)
        x_fast = self.reduce_fast(x_fast.squeeze()).reshape(-1,2048,1,1,1)
        x_slow = self.avg_pool(x_slow)
        if self.PCA:
            x_slow = x_slow.reshape(-1,4,4,4,4)
            x_fast = x_fast.reshape(-1, 4, 4, 4, 32)
            x_slow_ = x_slow# zero_topN_channels_by_info(x_slow,0,mode=self.mode)
            x_fast_ = x_fast# zero_topN_channels_by_info(x_fast,0,mode=self.mode)
            # print('NNNs')
            x_fast_ = x_fast_.reshape((-1,32,4,4,4))
            x_fast = x_fast.reshape((-1,32,4,4,4))
            # import pdb;pdb.set_trace()
        # [N, channel_fast + channel_slow, 1, 1, 1]
        ################################
        # We send our Feature Context to Server here // with x_fast 16,512,1,1,1 & x_slow 16,256,1,1,1
        if self.Enc:
            flat_fast = x_fast.view(-1).tolist()
            encrypted_fast = ts.ckks_vector(self.context, flat_fast)
            flat_slow = x_slow.view(-1).tolist()
            encrypted_slow = ts.ckks_vector(self.context, flat_slow)
            x_fast = encrypted_fast.view(-1, 2048, 1, 1, 1)
            x_slow = encrypted_slow.view(-1, 256, 1, 1, 1)

        if self.Hyper:
            kernel_fast = self.Hyper_fast(x_fast.squeeze())
            # import pdb;pdb.set_trace()
            kernel_slow = self.Hyper_slow(x_slow.squeeze())
            batch_size = x_fast.shape[0]
            in_channels = (x_fast.shape[1], x_slow.shape[1])
            out_channels = in_channels

            self.DCNN_slow = nn.Conv3d(in_channels=batch_size * in_channels[1],
                                       out_channels=batch_size * out_channels[1],
                                       kernel_size=(1, 1, 1),
                                       padding=0,
                                       groups=batch_size).cuda()
            self.DCNN_fast = nn.Conv3d(in_channels=batch_size * in_channels[0],
                                       out_channels=batch_size * out_channels[0],
                                       kernel_size=(1, 1, 1),
                                       padding=0,
                                       groups=batch_size).cuda()

            with torch.no_grad():
                self.DCNN_fast.weight.copy_(kernel_fast.reshape(-1, 2048, 1, 1, 1))
                # self.DCNN_fast.bias.copy_(predefined_bias)
                self.DCNN_slow.weight.copy_(kernel_slow.reshape(-1, 256, 1, 1, 1))
            x_fast_conv = self.DCNN_fast(x_fast.reshape(-1, 1, 1, 1))
            # x_fast_conv = self.DCNN_fast(x_fast.unsqueeze(2).unsqueeze(3), kernel_fast)
            x_slow_conv = self.DCNN_slow(x_slow.reshape(-1, 1, 1, 1))
            x_fast_conv = x_fast_conv.reshape(-1, 2048, 1, 1, 1)
            x_slow_conv = x_slow_conv.reshape(-1, 256, 1, 1, 1)
            x = torch.cat((x_slow_conv, x_fast_conv), dim=1)
        else:
            x = torch.cat((x_slow, x_fast), dim=1)
            if self.Test:
                x_= torch.cat((x_slow_, x_fast_), dim=1)
        ################################

        # x = torch.cat((x_slow, x_fast), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)
            if self.Test:
                x_ = self.dropout(x_)
        # [N x C]
        x = x.view(x.size(0), -1)
        if self.Test:
            x_ = x_.view(x.size(0), -1)

        if self.reduce_channel:
            x = self.fc_reduce(x)
            if self.Test:
                x_ = self.fc_reduce(x_)

        # [N x num_classes]
        # import pdb;pdb.set_trace()
        ################################
        # cls_score = self.fc_cls(x)
        # import pdb;pdb.set_trace()
        if self.Hyper:
            para, bias = self.Hyper_Cls(x)
            cls_score = (torch.bmm(x.unsqueeze(1),para).squeeze() + bias)
        else:
            cls_score = self.fc_cls(x)
            if self.Test:
                cls_score_ = self.fc_cls(x_)
        ################################
        if self.Enc:
            return cls_score, x, self.context
        elif self.Test:
            return cls_score, x, cls_score_, x_
        else:
            return cls_score, x

'''

'''
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class SlowFastHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 reduce_channel=False,
                 reduce_channel_num=2048,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.reduce_channel = reduce_channel
        self.reduce_channel_num = reduce_channel_num

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # if self.reduce_channel:
        #     print("reduce_channel")
        #     self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)
        #     self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)
        # else:
        #     self.fc_cls = nn.Linear(in_channels, num_classes)

        self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)
        self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        x_fast, x_slow = x
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        x_fast = self.avg_pool(x_fast)
        x_slow = self.avg_pool(x_slow)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        x = torch.cat((x_slow, x_fast), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        # [N x C]
        x = x.view(x.size(0), -1)

        if self.reduce_channel:
            x = self.fc_reduce(x)

        # [N x num_classes]
        # import pdb;pdb.set_trace()
        # x[:, :2304] = 0
        cls_score = self.fc_cls(x)

        return cls_score, x
'''
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import tenseal as ts
from ..registry import HEADS
from .base import BaseHead
from ..hypernetwork.hypernetwork_modules import HyperNetwork
from ..hypernetwork.hypernetwork_modules import HyperNetwork_Linear, HyperNetwork_Li, Embedding

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def zero_topN_channels_by_info(X: torch.Tensor, N: int, bins: int = 256, mode='Info') -> torch.Tensor:
    # 创建一个 X 的副本，不改变原来的 X
    X_mod = X.clone()
    assert X_mod.ndim == 5
    if mode == 'Info':
        B, T, H, W, C = X_mod.shape
        assert N <= C
        X_cpu = X_mod.cpu()
        entropies = []
        variances = []
        for c in range(C):
            channel_data = X_cpu[..., c].view(-1)
            var_c = channel_data.var().item()
            data_min = channel_data.min().item()
            data_max = channel_data.max().item()
            if data_min == data_max:
                ent_c = 0.0
            else:
                hist = torch.histc(channel_data, bins=bins, min=data_min, max=data_max)
                p = hist / hist.sum()
                p_nonzero = p[p > 0]
                ent_c = -(p_nonzero * p_nonzero.log()).sum().item()
            entropies.append(ent_c)
            variances.append(var_c)
        entropies = torch.tensor(entropies, dtype=torch.float32)
        variances = torch.tensor(variances, dtype=torch.float32)
        entropies_norm = torch.softmax(entropies, dim=0)
        variances_norm = torch.softmax(variances, dim=0)
        info_scores = 0.5 * entropies_norm + 0.5 * variances_norm
        scores_sorted, indices_sorted = torch.sort(info_scores, descending=True)
        # 反向遮掩是N:
        topN_indices = indices_sorted[N:]
        with torch.no_grad():
            for c_idx in topN_indices:
                X_mod[..., c_idx] = 0.0
    elif mode == 'RandCh':
        B, T, H, W, C = X_mod.shape
        num_to_zero = N
        random_indices = torch.randperm(C)[:num_to_zero]
        with torch.no_grad():
            for c_idx in random_indices:
                X_mod[..., c_idx] = 0.0
    elif mode == 'Rand Half':
        numel = X_mod.numel()
        B, T, H, W, C = X_mod.shape
        num_to_zero = numel // C * N
        X_flat = X_mod.view(-1)
        indices = torch.randperm(numel)[:num_to_zero]
        with torch.no_grad():
            X_flat[indices] = 0.0
    else:
        pass
    return X_mod

print_frag = False

@HEADS.register_module()
class SlowFastHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 reduce_channel=False,
                 reduce_channel_num=2048,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.reduce_channel = reduce_channel
        self.reduce_fast = nn.Linear(reduce_channel_num, reduce_channel_num)
        self.reduce_channel_num = reduce_channel_num
        self.Hyper = False  # Use adaptive cnn or not
        self.PCA = True  # Training false/ Test True
        self.Enc = False and self.PCA  # Keys
        self.Test = self.PCA  # Training False Test True
        if self.Hyper:
            self.Hyper_slow = HyperNetwork(f_size=1, z_dim=256, out_size=256, in_size=256)
            self.Hyper_fast = HyperNetwork(f_size=1, z_dim=512, out_size=512, in_size=512)
            self.Hyper_Cls = HyperNetwork_Li(hidden=32, z_dim=768, label_size=25, batch_size=16, zoom_dim=768)
        if self.Enc:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, [60, 40, 40, 60])
            self.context.global_scale = 2 ** 40
            self.context.generate_galois_keys()
        if self.PCA:
            self.mode = 'RandCh'
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # if self.reduce_channel:
        #     print("reduce_channel")
        #     self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)
        #     self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)
        # else:
        #     self.fc_cls = nn.Linear(in_channels, num_classes)

        self.fc_reduce = nn.Linear(in_channels, self.reduce_channel_num)

        # self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)
        self.fc_cls = nn.Linear(self.reduce_channel_num, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        x_fast, x_slow = x
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace(   )
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        x_fast = self.avg_pool(x_fast)
        x_fast = self.reduce_fast(x_fast.squeeze()).reshape(-1, 2048, 1, 1, 1)
        x_slow = self.avg_pool(x_slow)
        if self.PCA:
            x_slow = x_slow.reshape(-1, 4, 4, 4, 4)
            x_fast = x_fast.reshape(-1, 4, 4, 4, 32)
            #import pdb;pdb.set_trace()
            slow_block_rate = 0.5
            fast_block_rate = slow_block_rate
            if print_frag:
                print('We are conducting zero channel by:'+self.mode)
                print('slow_block_rate:'+str(slow_block_rate))
                print('fast_block_rate:'+str(fast_block_rate))
            print_flag = False
            x_slow_ = zero_topN_channels_by_info(x_slow, int(4 * slow_block_rate), mode=self.mode)
            x_fast_ = zero_topN_channels_by_info(x_fast, int(32 *fast_block_rate), mode=self.mode)
            x_fast_ = x_fast_.reshape((-1, 32, 4, 4, 4))
            x_fast = x_fast.reshape((-1, 32, 4, 4, 4))
            # import pdb;pdb.set_trace()
        # [N, channel_fast + channel_slow, 1, 1, 1]
        ################################
        # We send our Feature Context to Server here // with x_fast 16,512,1,1,1 & x_slow 16,256,1,1,1
        if self.Enc:
            flat_fast = x_fast.view(-1).tolist()
            encrypted_fast = ts.ckks_vector(self.context, flat_fast)
            flat_slow = x_slow.view(-1).tolist()
            encrypted_slow = ts.ckks_vector(self.context, flat_slow)
            x_fast = encrypted_fast.view(-1, 2048, 1, 1, 1)
            x_slow = encrypted_slow.view(-1, 256, 1, 1, 1)

        if self.Hyper:
            kernel_fast = self.Hyper_fast(x_fast.squeeze())
            # import pdb;pdb.set_trace()
            kernel_slow = self.Hyper_slow(x_slow.squeeze())
            batch_size = x_fast.shape[0]
            in_channels = (x_fast.shape[1], x_slow.shape[1])
            out_channels = in_channels

            self.DCNN_slow = nn.Conv3d(in_channels=batch_size * in_channels[1],
                                       out_channels=batch_size * out_channels[1],
                                       kernel_size=(1, 1, 1),
                                       padding=0,
                                       groups=batch_size).cuda()
            self.DCNN_fast = nn.Conv3d(in_channels=batch_size * in_channels[0],
                                       out_channels=batch_size * out_channels[0],
                                       kernel_size=(1, 1, 1),
                                       padding=0,
                                       groups=batch_size).cuda()

            with torch.no_grad():
                self.DCNN_fast.weight.copy_(kernel_fast.reshape(-1, 2048, 1, 1, 1))
                # self.DCNN_fast.bias.copy_(predefined_bias)
                self.DCNN_slow.weight.copy_(kernel_slow.reshape(-1, 256, 1, 1, 1))
            x_fast_conv = self.DCNN_fast(x_fast.reshape(-1, 1, 1, 1))
            # x_fast_conv = self.DCNN_fast(x_fast.unsqueeze(2).unsqueeze(3), kernel_fast)
            x_slow_conv = self.DCNN_slow(x_slow.reshape(-1, 1, 1, 1))
            x_fast_conv = x_fast_conv.reshape(-1, 2048, 1, 1, 1)
            x_slow_conv = x_slow_conv.reshape(-1, 256, 1, 1, 1)
            x = torch.cat((x_slow_conv, x_fast_conv), dim=1)
        else:
            x = torch.cat((x_slow, x_fast), dim=1)
            if self.Test:
                x_ = torch.cat((x_slow_, x_fast_), dim=1)
        ################################

        # x = torch.cat((x_slow, x_fast), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)
            if self.Test:
                x_ = self.dropout(x_)
        # [N x C]
        x = x.view(x.size(0), -1)
        if self.Test:
            x_ = x_.view(x.size(0), -1)

        if self.reduce_channel:
            x = self.fc_reduce(x)
            if self.Test:
                x_ = self.fc_reduce(x_)

        # [N x num_classes]
        # import pdb;pdb.set_trace()
        ################################
        # cls_score = self.fc_cls(x)
        # import pdb;pdb.set_trace()
        if self.Hyper:
            para, bias = self.Hyper_Cls(x)
            cls_score = (torch.bmm(x.unsqueeze(1), para).squeeze() + bias)
        else:
            cls_score = self.fc_cls(x)
            if self.Test:
                cls_score_ = self.fc_cls(x_)
        ################################
        if self.Enc:
            return cls_score, x, self.context
        elif self.Test:
            # import pdb;pdb.set_trace()
            return cls_score, x, cls_score_, x_
        else:
            return cls_score, x # , cls_score, x
