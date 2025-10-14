"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
import torch.nn.functional as F

try:
    #import flash_attn
    import flash_attn.flash_attn_interface as flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_unpadded_qkvpacked_func(
            #feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            target=point.target,
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # For 2D Tocken.
        self.return_interm_indices = [1, 2, 3]  # args.return_interm_indices
        assert self.return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def project_pointcloud_to_fixed_grid(self, points, features, r, H, W, x_min, y_min):
        """
        Project a point cloud onto the x-y plane with resolution r and fixed grid size.
        Each grid cell stores the AVERAGE feature of points inside it.
        Empty cells remain zeros.

        Args:
            points (torch.Tensor): (N, 3) tensor of point coordinates (x, y, z).
            features (torch.Tensor): (N, c) tensor of point features.
            r (float): resolution (grid cell size).
            H (int): grid height (#cells along y).
            W (int): grid width (#cells along x).

        Returns:
            grid_features (torch.Tensor): (H, W, c) feature map.
            x_edges (torch.Tensor): (W+1,) bin edges along x-axis.
            y_edges (torch.Tensor): (H+1,) bin edges along y-axis.
        """
        assert points.shape[0] == features.shape[0]

        device = points.device
        N, C = features.shape

        # Extract x, y
        x, y = points[:, 0], points[:, 1]

        # Center the grid on the point cloud
        #x_center, y_center = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        #x_min, x_max = x_center - W * r / 2, x_center + W * r / 2
        #y_min, y_max = y_center - H * r / 2, y_center + H * r / 2

        # Compute grid indices
        ix = ((x - x_min) / r).long()
        iy = ((y - y_min) / r).long()

        # Filter valid indices (inside grid)
        mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy, feats = ix[mask], iy[mask], features[mask]

        # Linear index for scatter ops
        linear_idx = iy * W + ix

        # Accumulate sums with scatter_add
        grid_sum = torch.zeros((H * W, C), device=device, dtype=features.dtype)
        grid_count = torch.zeros((H * W, 1), device=device, dtype=features.dtype)

        grid_sum.index_add_(0, linear_idx, feats)
        grid_count.index_add_(0, linear_idx, torch.ones((linear_idx.shape[0], 1), device=device, dtype=features.dtype))

        # Avoid division by zero: empty cells stay zero
        grid_features = torch.where(grid_count > 0, grid_sum / grid_count, torch.zeros_like(grid_sum))

        # Reshape back to (H, W, C)
        grid_features = grid_features.view(H, W, C)

        # Bin edges
        #x_edges = torch.linspace(x_min, x_max, W + 1, device=device)
        #y_edges = torch.linspace(y_min, y_max, H + 1, device=device)

        return grid_features

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)

        #point = self.enc(point)
        # tokens, enc_point = self.enc(point) # should be split into each layer to extract tokens
        # Should be split by offset!!!!!!!!!!!!!!!!!!!!!!!!!
        tokens = []

        import numpy as np
        import cv2
        from math import pi, cos, sin
        def draw_obb_on_tensor(image_tensor, obbs, color=(0, 255, 0), thickness=2):
            """
            Draw oriented bounding boxes on an image tensor.

            Args:
                image_tensor (torch.Tensor): (H, W, C), values in [0,1] or [0,255]
                obbs (torch.Tensor): (N, 5), each row = (cx, cy, w, h, theta_norm)
                color (tuple): BGR color for drawing
                thickness (int): line thickness

            Returns:
                img_out (np.ndarray): image with OBBs drawn (uint8, BGR)
            """
            H, W, C = image_tensor.shape

            # tensor -> numpy (uint8, BGR)
            img = image_tensor.detach().cpu().numpy()
            img /= img.max()
            if img.max() <= 1.0:  # [0,1] -> [0,255]
                img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for cx, cy, w, h, theta_n in obbs.cpu().numpy():
                # 1. 디노멀라이즈
                cx, cy = cx * W, cy * H
                w, h = w * W, h * H
                theta = (theta_n - 0.5) * pi  # 라디안

                # 2. 회전 박스 꼭짓점 계산
                cos_t, sin_t = cos(theta), sin(theta)
                dx, dy = w / 2, h / 2
                corners = np.array([
                    [-dx, -dy],
                    [dx, -dy],
                    [dx, dy],
                    [-dx, dy]
                ])

                R = np.array([[cos_t, -sin_t],
                              [sin_t, cos_t]])
                rotated = corners @ R.T
                rotated[:, 0] += cx
                rotated[:, 1] += cy

                # 3. OpenCV용 integer 좌표
                pts = rotated.astype(np.int32).reshape((-1, 1, 2))

                # 4. 이미지에 그리기
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

            return img

        def pointcloud_feature_grid(grid_x, grid_y, grid_feat, size, grid_size=3.0, device=None):
            """
            Project a point cloud with feature vectors to a 2D grid and compute average per-cell features (differentiable).

            Args:
                grid_x, grid_y: (N,) torch.Tensor, point coordinates
                grid_feat: (N, C) torch.Tensor, feature vectors per point
                H, W: original target size
                grid_size: size of each grid cell
                device: torch device (optional)

            Returns:
                grid_mean: (C, ny, nx) torch.Tensor, average feature map per grid cell
                grid_count: (1, ny, nx) torch.Tensor, number of points per cell
            """

            if device is None:
                device = grid_x.device

            N, C = grid_feat.shape

            # -----------------------------
            # 1. Determine grid size
            # -----------------------------
            #nx = max(math.ceil(H / grid_size), int(grid_x.max().item()) + 1)
            #ny = max(math.ceil(W / grid_size), int(grid_y.max().item()) + 1)
            nx = ny = size # ensure square grid

            # -----------------------------
            # 2. Convert to integer indices
            # -----------------------------
            ix = grid_x.long()
            iy = grid_y.long()

            # -----------------------------
            # 3. Initialize sum and count tensors
            # -----------------------------
            grid_sum = torch.zeros((C, ny, nx), dtype=grid_feat.dtype, device=device)
            grid_count = torch.zeros((1, ny, nx), dtype=grid_feat.dtype, device=device)

            # -----------------------------
            # 4. Scatter-add features per channel
            # -----------------------------
            for c in range(C):
                grid_sum[c].index_put_((iy, ix), grid_feat[:, c], accumulate=True)

            grid_count[0].index_put_((iy, ix),
                                     torch.ones_like(ix, dtype=grid_feat.dtype),
                                     accumulate=True)

            # -----------------------------
            # 5. Compute average per cell
            # -----------------------------
            grid_mean = grid_sum / (grid_count + 1e-6)

            return grid_mean # C, H, W


        for s in range(self.num_stages):
            point = self.enc[s](point)

            start_point = 0
            grid_size = 3 * (2**s)  ##<- should be an arg
            grid_size = torch.tensor(grid_size, device=point.coord.device, dtype=point.coord.dtype)
            for i, offset in enumerate(point.offset):
                coord = point.coord[start_point:offset]
                H, W = point.target[i]['size']

                minx = int((W + coord[:, 0].min()) / (grid_size)) if point.target[i]["hflip"] else int(
                    coord[:, 0].min() / grid_size)
                miny = int((H + coord[:, 1].min()) / (grid_size)) if point.target[i]["vflip"] else int(
                    coord[:, 1].min() / grid_size)

                grid_coord = point.grid_coord[start_point:offset]


                grid_x = grid_coord[:, 0] + minx
                grid_y = grid_coord[:, 1] + miny

                # rescale grid_x and grid_y to make them the same size.
                nx = max(math.ceil(H / grid_size), int(grid_x.max().item()))
                ny = max(math.ceil(W / grid_size), int(grid_y.max().item()))

                size = math.ceil(1024 / grid_size)
                scale_x = size / nx
                scale_y = size / ny

                grid_x = grid_x * scale_x
                grid_y = grid_y * scale_y

                if grid_x.max() == size:
                    idx = (grid_x == size).nonzero(as_tuple=True)[0]  # index of max
                    grid_x[idx] = size - 1
                if grid_y.max() == size :
                    idx = (grid_y == size).nonzero(as_tuple=True)[0]  # index of max
                    grid_y[idx] = size - 1

                print (math.ceil(1024/grid_size), grid_x.max(), grid_y.max())

                grid_feat = grid_coord[:, 2:].repeat(1, 3)

                grid_z = pointcloud_feature_grid(grid_x, grid_y, grid_feat, size, grid_size)
                print(i, s, grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max(), grid_z.shape)

                grid_z = grid_z.permute(1, 2, 0)
                obbs = point.target[i]['boxes']

                img_with_boxes = draw_obb_on_tensor(grid_z, obbs)
                cv2.imwrite("/nas2/YJ/git/Fusion/exp/debug/batch_{}_enc_{}output.png".format(i, s), img_with_boxes)

                # vutils.save_image(grid_z.permute(2, 0, 1), "/nas2/YJ/git/Fusion/exp/debug/batch_{}_output.png".format(i))
                start_point = offset

            '''
            if s in self.return_interm_indices:
                #tokens.append(point)  ###<------------ should be list of tokens which element's shape is [B, C, H, W] with mask(False mean real value)
                #grid_coord = point.grid_coord
                #print (s, grid_coord.min(0).values, grid_coord.max(0).values)
                #grid_size = 3. ###<-------- should be an arg
                #scaled_coord = data_dict["coord"] / np.array(grid_size)
                #grid_coord = np.floor(scaled_coord).astype(int)
                #min_coord = grid_coord.min(0)

                #grid_coord -= min_coord

                r = 2 ** s
                H = W = 1024
                coord = point.coord
                feats = point.feat
                z_feats = coord[:, 2:].repeat(1, 3)
                grid = self.project_pointcloud_to_fixed_grid(coord, feats, r, H, W, -512, -512)
                grid_z = self.project_pointcloud_to_fixed_grid(coord, z_feats, r, H, W, -512, -512)
                #### We can't use average pooling here!!. We need to use grid_coord directly!!!!
                #feat = F.avg_pool2d(grid.permute(2, 0, 1).unsqueeze(0), kernel_size=r, stride=r)[0]
                #feat_z = F.avg_pool2d(grid_z.permute(2, 0, 1).unsqueeze(0), kernel_size=r, stride=r)[0]

                import torchvision.utils as vutils
                import cv2
                vutils.save_image(feat_z, "/nas2/YJ/git/Fusion/exp/debug/output_{}.png".format(s))
                mask = (feat_z == 0).all(dim=0)
                cv2.imwrite("/nas2/YJ/git/Fusion/exp/debug/mask_{}.png".format(s), ((feat_z == 0).all(dim=0).to(torch.uint8) * 255).cpu().numpy())
                print (s, mask.sum(), mask.shape, mask.sum()/(mask.shape[0]*mask.shape[1]))
            '''



        if not self.cls_mode:
            point = self.dec(point)

        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point
