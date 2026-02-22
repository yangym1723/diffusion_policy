from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.history_action_encoder import HistoryActionEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.shape_util import get_output_shape
from diffusion_policy.model.obs_encoder.temporal_aggregator import TemporalAggregator


class DiffusionUnetVideoPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            rgb_net: nn.Module, # (B,T,C,H,W) -> (B,Do)
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            lowdim_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters for TemporalAggregator
            channel_mults=(1,1),
            n_blocks_per_level=1,
            ta_kernel_size=3,
            ta_n_groups=8,
            use_history_encoder=False,
            history_encoder_hidden_dim=128,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']

        rgb_nets_map = nn.ModuleDict()
        rgb_feature_dims = list()
        lowdim_keys = list()
        lowdim_input_dims = list()

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'lowdim')
            if type == 'rgb':
                # assign network for each rgb input
                if len(rgb_nets_map) == 0:
                    net = rgb_net
                else:
                    net = copy.deepcopy(rgb_net)
                rgb_nets_map[key] = net

                # video input with n_obs_steps timesteps
                shape = (n_obs_steps,) + shape
                # compute output shape
                output_shape = get_output_shape(shape, net)
                assert(len(output_shape) == 1)
                rgb_feature_dims.append(output_shape[0])
            elif type == 'lowdim':
                lowdim_keys.append(key)
                assert(len(shape) == 1)
                lowdim_input_dims.append(shape[0])

        # the order decides concatenation order
        # dict preserves insertion order
        # rgb and then lowdim
        self.rgb_nets_map = rgb_nets_map
        self.lowdim_keys = lowdim_keys
        self.lowdim_net = None

        # compute dimensions for diffusion
        rgb_feature_dim = sum(rgb_feature_dims)
        lowdim_input_dim = sum(lowdim_input_dims)
        global_cond_dim = rgb_feature_dim
        input_dim = action_dim
        if lowdim_as_global_cond:
            lowdim_net = TemporalAggregator(
                in_channels=lowdim_input_dim,
                channel_mults=channel_mults,
                n_blocks_per_level=n_blocks_per_level,
                kernel_size=ta_kernel_size,
                n_groups=ta_n_groups
            )
            self.lowdim_net = lowdim_net
            lowdim_feature_shape = get_output_shape(
                (n_obs_steps, lowdim_input_dim), lowdim_net)
            assert len(lowdim_feature_shape) == 1
            global_cond_dim += lowdim_feature_shape[0]
        else:
            input_dim += lowdim_input_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if lowdim_as_global_cond else lowdim_input_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.lowdim_input_dim = lowdim_input_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.lowdim_as_global_cond = lowdim_as_global_cond
        self.use_history_encoder = use_history_encoder
        self.kwargs = kwargs
        self.history_encoder = None
        if self.use_history_encoder:
            self.history_encoder = HistoryActionEncoder(
                action_dim=action_dim,
                output_dim=lowdim_input_dim,
                hidden_dim=history_encoder_hidden_dim
            )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _shifted_prefix_sum(self, action_seq: torch.Tensor) -> torch.Tensor:
        prefix = torch.cumsum(action_seq, dim=1)
        zero = torch.zeros_like(prefix[:, :1])
        return torch.cat([zero, prefix[:, :-1]], dim=1)

    def _prefix_sum(self, action_seq: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(action_seq, dim=1)

    def _align_steps(self, x: torch.Tensor, target_steps: int) -> torch.Tensor:
        bsz, steps, dim = x.shape
        if steps == target_steps:
            return x
        if steps > target_steps:
            return x[:, -target_steps:]
        pad = torch.zeros(
            size=(bsz, target_steps - steps, dim),
            device=x.device,
            dtype=x.dtype
        )
        return torch.cat([pad, x], dim=1)

    def _encode_history(self, action_seq: torch.Tensor, target_steps: int, shifted: bool) -> torch.Tensor:
        history = self._shifted_prefix_sum(action_seq) if shifted else self._prefix_sum(action_seq)
        history = self._align_steps(history, target_steps)
        return self.history_encoder(history)
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        obs_input = {k: v for k, v in obs_dict.items() if k != 'past_action'}
        nobs = self.normalizer.normalize(obs_input)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # run encoder first
        # python 3.6+ dict preserves order
        rgb_features_map = dict()
        for key, net in self.rgb_nets_map.items():
            rgb_features_map[key] = net(nobs[key][:,:self.n_obs_steps])
        rgb_feature = torch.cat(list(rgb_features_map.values()), dim=-1)

        lowdim_input = torch.cat([nobs[k] for k in self.lowdim_keys], dim=-1)
        if self.use_history_encoder:
            history_feat = torch.zeros_like(lowdim_input)
            if ('past_action' in obs_dict) and (obs_dict['past_action'] is not None):
                npast_action = self.normalizer['action'].normalize(obs_dict['past_action'])
                history_feat = self._encode_history(npast_action, lowdim_input.shape[1], shifted=False)
            lowdim_input = lowdim_input + history_feat

        # handle different ways of passing lowdim
        global_cond = None
        cond_data = None
        cond_mask = None
        if self.lowdim_as_global_cond:
            lowdim_feature = self.lowdim_net(lowdim_input[:,:To])
            global_cond = torch.cat([rgb_feature, lowdim_feature], dim=-1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            global_cond = rgb_feature
            cond_data = torch.zeros(size=(B, T, Da+self.lowdim_input_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = lowdim_input[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # run encoder first
        # python 3.6+ dict preserves order
        rgb_features_map = dict()
        for key, net in self.rgb_nets_map.items():
            rgb_features_map[key] = net(nobs[key][:,self.n_obs_steps:])
        rgb_feature = torch.cat(list(rgb_features_map.values()), dim=-1)

        lowdim_input = torch.cat([nobs[k] for k in self.lowdim_keys], axis=-1)
        if self.use_history_encoder:
            lowdim_input = lowdim_input + self._encode_history(
                nactions, lowdim_input.shape[1], shifted=True)
        
        # handle different ways of passing lowdim
        global_cond = None
        trajectory = None
        cond_data = None
        if self.lowdim_as_global_cond:
            lowdim_feature = self.lowdim_net(lowdim_input[:,:self.n_obs_steps])
            global_cond = torch.cat([rgb_feature, lowdim_feature], dim=-1)
            trajectory = nactions
            cond_data = nactions
        else:
            global_cond = rgb_feature
            trajectory = torch.cat([nactions, lowdim_input], dim=-1)
            cond_data = trajectory

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond)

        if self.kwargs.get('predict_epsilon', True):
            # default for most methods
            target = noise
        else:
            # DDPM also has
            target = trajectory

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
