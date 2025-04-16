import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, T5EncoderModel

from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DConditionGuidedModel, HeunDiscreteScheduler
from audioldm.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config


class ConsistencyTTA(nn.Module):









    @torch.no_grad()


    @torch.no_grad()



        # Query the consistency model
        zhat_0 = calc_zhat_0(z_N, self.scheduler.timesteps[0])

        # Iteratively query the consistency model if requested
        self.scheduler.set_timesteps(num_steps, device=device)

        for t in self.scheduler.timesteps[1::2]:  # 2 is the order of the scheduler
            zhat_n = self.scheduler.add_noise(zhat_0, torch.randn_like(zhat_0), t)
            # Calculate new zhat_0
            zhat_0 = calc_zhat_0(zhat_n, t)

        mel = self.vae.decode_first_stage(zhat_0.float())
        return self.vae.decode_to_waveform(mel)[:, :int(sr * 9.5)]  # Truncate to 9.6 seconds