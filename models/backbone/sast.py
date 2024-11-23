import torch 
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from typing import Dict, Optional, Tuple

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..blocks.sast_stage import SASTStage

@torch.no_grad()
def non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
    mask = x.float() 
    x_down_4 = torch.nn.functional.max_pool2d(mask, kernel_size=4, stride=4)
    x_down_8 = torch.nn.functional.max_pool2d(x_down_4, kernel_size=2, stride=2)
    x_down_16 = torch.nn.functional.max_pool2d(x_down_8, kernel_size=2, stride=2)
    x_down_32 = torch.nn.functional.max_pool2d(x_down_16, kernel_size=2, stride=2)
    num_nonzero_1 = torch.sum(torch.sum(x_down_4 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_2 = torch.sum(torch.sum(x_down_8 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_3 = torch.sum(torch.sum(x_down_16 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_4 = torch.sum(torch.sum(x_down_32 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    result1 = x.shape[0] / x_down_4.numel() * num_nonzero_1.float()
    result2 = x.shape[0] / x_down_8.numel() * num_nonzero_2.float()
    result3 = x.shape[0] / x_down_16.numel() * num_nonzero_3.float()
    result4 = x.shape[0] / x_down_32.numel() * num_nonzero_4.float()
    return torch.stack((result1, result2, result3, result4), dim=1)

class SAST(nn.Module):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
        enable_masking = mdl_config.enable_masking

        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)


        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################

        input_dim = in_channels
        patch_size = mdl_config.stem.patch_size
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.stages = nn.ModuleList()
        self.strides = []
        in_res_h, in_res_w = mdl_config.in_res_hw
        initial_size = (1, in_res_h, in_res_w)
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            overload_size = (1, initial_size[1] // spatial_downsample_factor, initial_size[2] // spatial_downsample_factor)
            enable_lstm = stage_idx != 0
            initial_size = overload_size
            stage = SASTStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config.stage,
                                     overload_size=overload_size,
                                     enable_lstm=True)
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
            self.stages.append(stage)

        self.num_stages = num_stages


    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    def forward(self, x: torch.Tensor, prev_states = None, token_mask: Optional[torch.Tensor] = None) \
:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states = list()
        output = {}

        r = non_zero_ratio(x)
        x = x.float()

        P = []
        for stage_idx, stage in enumerate(self.stages):
            x, state, p = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None, r[:, stage_idx])
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = state[0]
            P.append(p)
        return output, states, P