import math
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Tuple, List

from ..layers.downsampling import get_downsample_layer_Cf2Cl, nhwC_2_nChw
from .lstm import DWSConvLSTM2d
from .SAST_block import SASTAttentionPairCl, PositionEmbeddingSine
class SASTStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig,
                 overload_size: Tuple[int, int, int],
                 enable_lstm: bool):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [SASTAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg, first_block=i == 0) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)
        self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0)) if enable_lstm else None

        self.pos_emb = PositionEmbeddingSine(stage_dim // 2, normalize=True, input_size=overload_size)

        ###### Mask Token ################
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)
        # self.weights = nn.Parameter(torch.ones(2), requires_grad=True)
        ##################################

    def forward(self, x: torch.Tensor,
                h_and_c_previous = None,
                token_mask: Optional[torch.Tensor] = None, r: torch.Tensor = None):
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token

        P = 0
        index_list = None
        for blk in self.att_blocks:
            x, p_loss, r, index_list = blk(x, self.pos_emb, r, index_list)
            P += p_loss
            
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        if self.lstm is not None:
            h_c_tuple = self.lstm(x, h_and_c_previous)
            x = h_c_tuple[0]
        else:
            h_c_tuple = (x, x)
        return x, h_c_tuple, P