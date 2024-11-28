import torch as th
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple
from einops import rearrange

from .lstm import DWSConvLSTM2d
from ..s5.s5_model import S5Block
from .attention import MaxVitAttentionPairCl
from ..layers.downsampling import get_downsample_layer_Cf2Cl, nhwC_2_nChw

class RVTStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init ,
                 stage_cfg: DictConfig):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)
        self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################

    def forward(self, x: th.Tensor,
                h_and_c_previous = None,
                token_mask = None):
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token
        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple
    
class RVTSSMStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output."""

    def __init__(
        self,
        dim_in: int,
        stage_dim: int,
        spatial_downsample_factor: int,
        num_blocks: int,
        enable_token_masking: bool,
        T_max_chrono_init: Optional[int],
        stage_cfg: DictConfig,
    ):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        # lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(
            dim_in=dim_in,
            dim_out=stage_dim,
            downsample_factor=spatial_downsample_factor,
            downsample_cfg=downsample_cfg,
        )
        blocks = [
            MaxVitAttentionPairCl(
                dim=stage_dim,
                skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                attention_cfg=attention_cfg,
            )
            for i in range(num_blocks)
        ]
        self.att_blocks = nn.ModuleList(blocks)

        self.s5_block = S5Block(
            dim=stage_dim, state_dim=stage_dim, bidir=False, bandlimit=0.5
        )

        """
        self.lstm = DWSConvLSTM2d(
            dim=stage_dim,
            dws_conv=lstm_cfg.dws_conv,
            dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
            dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
            cell_update_dropout=lstm_cfg.get("drop_cell_update", 0),
        )
        """

        ###### Mask Token ################
        self.mask_token = (
            nn.Parameter(th.zeros(1, 1, 1, stage_dim), requires_grad=True)
            if enable_token_masking
            else None
        )

        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=0.02)
        ##################################

    def forward(
        self,
        x: th.Tensor,
        states = None,
        token_mask: Optional[th.Tensor] = None,
        train_step: bool = True,):

        sequence_length = x.shape[0]
        batch_size = x.shape[1]
        x = rearrange(
            x, "L B C H W -> (L B) C H W"
        )  # where B' = (L B) is the new batch size
        x = self.downsample_cf2cl(x)  # B' C H W -> B' H W C

        if token_mask is not None:
            assert self.mask_token is not None, "No mask token present in this stage"
            x[token_mask] = self.mask_token

        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)  # B' H W C -> B' C H W

        new_h, new_w = x.shape[2], x.shape[3]

        x = rearrange(x, "(L B) C H W -> (B H W) L C", L=sequence_length)

        if states is None:
            states = self.s5_block.s5.initial_state(
                batch_size=batch_size * new_h * new_w
            ).to(x.device)
        else:
            states = rearrange(states, "B C H W -> (B H W) C")

        x, states = self.s5_block(x, states)

        x = rearrange(
            x, "(B H W) L C -> L B C H W", B=batch_size, H=int(new_h), W=int(new_w)
        )

        states = rearrange(states, "(B H W) C -> B C H W", H=new_h, W=new_w)

        return x, states