import glob
import os
import pathlib
import argparse
from typing import Union, Tuple, List
from joblib import Parallel, delayed
from functools import partial
from dataclasses import dataclass, field
from math import ceil, sqrt

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, utils
# from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from accelerate import Accelerator
from accelerate.utils import LoggerType
import wandb
from matplotlib import pyplot as plt
from safetensors.torch import save_file, safe_open
import json 
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from diffusers import ConsistencyModelPipeline, CMStochasticIterativeScheduler
from util import list_all_images, set_generator, normalize

class MultiDataset(Dataset):
    REPEAT_BY_APPEND: str = "REPEAT_BY_APPEND"
    REPEAT_BY_INPLACE: str = "REPEAT_BY_INPLACE"
    
    NOISE_MAP_SAME: str = "NOISE_MAP_SAME"
    NOISE_MAP_DIFF: str = "NOISE_MAP_DIFF"
    
    CHANNEL_LAST: int = -1
    CHANNEL_FIRST: int = -3
    
    def __init__(self, src: Union[torch.Tensor, str, os.PathLike, pathlib.PurePath, List[Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]]], 
                 size: Union[int, Tuple[int, int], list[int]], 
                 repeat: int = 1, 
                 repeat_type: str=REPEAT_BY_INPLACE, 
                 noise_map_type: str=NOISE_MAP_SAME, 
                 vmin_out: float=-1.0, 
                 vmax_out: float=1.0, 
                 generator: Union[int, torch.Generator]=0, 
                 image_exts: list[str]=['png', 'jpg', 'jpeg', 'webp'], 
                 ext_case_sensitive: bool=False, 
                 njob: int=-1):
        if isinstance(src, list):
            self.__src__ = src
        elif isinstance(src, torch.Tensor):
            self.__src__ = torch.split(src, 1)
        elif isinstance(src, str) or isinstance(src, os.PathLike) or isinstance(src, pathlib.PurePath):
            self.__src__ = list_all_images(root=src, image_exts=image_exts, ext_case_sensitive=ext_case_sensitive)
        else:
            raise TypeError(f"Arguement src should not be {type(src)}, it should be Union[torch.Tensor, str, os.PathLike, pathlib.PurePath, List[Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]]]")
        
        # print(f"self.__src__: {len(self.__src__)}")
        
        self.__repeat__: int = repeat
        self.__repeat_type__: str = repeat_type
        self.__noise_map_type__: str = noise_map_type
        self.__generator__ = set_generator(generator=generator)
        self.__vmin_out__: float = vmin_out
        self.__vmax_out__: float = vmax_out
        self.__size__: Union[int, Tuple[int], List[int]] = size
        self.__njob__: int = njob
        self.__unique_src_n__: int = len(self)
        
        self.__build_src_noise__()
        
    def __repeat_data__(self, data: List):
        new_ls: List = []
        if self.__repeat__ > 1:
            if self.__repeat_type__ == MultiDataset.REPEAT_BY_INPLACE:
                for e in data:
                    new_ls = new_ls + [e] * self.__repeat__
                return new_ls
            elif self.__repeat_type__ == MultiDataset.REPEAT_BY_APPEND:
                return data * self.__repeat__
            else:
                raise ValueError(f"No such repeat type, {self.__repeat_type__}, should be {MultiDataset.REPEAT_BY_INPLACE} or {MultiDataset.REPEAT_BY_APPEND}")
        return data
    
    def __build_noise__(self):
        element_shape: List[int] = list(self.__loader__(data=self.__src__[0]).shape)
        if self.__noise_map_type__ == MultiDataset.NOISE_MAP_SAME:
            self.__noise__ = [x.squeeze(dim=0) for x in torch.randn(size=[self.__unique_src_n__] + element_shape, generator=self.__generator__).split(1)]
            self.__noise__ = self.__repeat_data__(data=self.__noise__)
        elif self.__noise_map_type__ == MultiDataset.NOISE_MAP_DIFF:
            self.__noise__ = [x.squeeze(dim=0) for x in torch.randn(size=[self.__unique_src_n__ * self.__repeat__] + element_shape, generator=self.__generator__).split(1)]
        else:
            raise ValueError(f"Arguement noise_map_type is {self.__noise_map_type__}, should be {MultiDataset.NOISE_MAP_SAME} or {MultiDataset.NOISE_MAP_DIFF}")
        
    def __build_src_noise__(self):
        self.__build_noise__()
        self.__src_idx__ = self.__repeat_data__(data=[i for i in range(self.__unique_src_n__)])
        self.__src__ = self.__repeat_data__(data=self.__src__)
    
    def __loader__(self, data: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]):
        trans = [
            transforms.Resize(size=self.__size__),
            transforms.ConvertImageDtype(torch.float),
            self.__forwar_reverse_fn__()[0],
        ]
        
        if isinstance(data, Image.Image):
            trans = [transforms.ToTensor()] + trans
        elif isinstance(data, str) or isinstance(data, os.PathLike) or isinstance(data, pathlib.PurePath):
            opening_image = transforms.Lambda(lambda x: Image.open(x).convert('RGB'))
            trans = [opening_image, transforms.ToTensor()] + trans
        else:
            raise TypeError(f"The arguement data should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}")
        
        return transforms.Compose(trans)(data)

    def __len__(self):
        return len(self.__src__)
    
    def to_tensor(self):
        res: List = list(self[0:len(self)])
        for i in range(len(res)):
            if isinstance(res[i][0], torch.Tensor):
                res[i] = torch.stack(res[i], dim=0)
            elif isinstance(res[i][0], int) or isinstance(res[i][0], float):
                res[i] = torch.tensor(res[i])
        return res

    def __getitem__(self, idx):
        src_ls: List = self.__src__[idx]
        if isinstance(src_ls, list):
            # return Parallel(n_jobs=self.__njob__)(delayed(self.__loader__)(e) for e in self.__src__[idx])
            return [self.__loader__(data=data) for data in src_ls], self.__noise__[idx], self.__src_idx__[idx]
            # return [self.__loader__(data=data) for data in src_ls]
        return self.__loader__(data=src_ls), self.__noise__[idx], self.__src_idx__[idx]
        # return self.__loader__(data=src_ls)
        
    def __forwar_reverse_fn__(self):
        norm_fn = [
            transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin_out__, vmax_out=self.__vmax_out__, x=x)),
            transforms.Lambda(lambda x: normalize(vmin_in=self.__vmin_out__, vmax_in=self.__vmax_out__, vmin_out=0, vmax_out=1, x=x))
        ]
        arithm_fn = [
            transforms.Lambda(lambda x: (x - 0.5) * 2),
            transforms.Lambda(lambda x: x / 2 + 0.5),
        ]
        torch_norm_fn = [
            transforms.Normalize(mean=0, std=0.5),
            transforms.Lambda(lambda x: x / 2 + 0.5),
        ]
        return arithm_fn
    
    def __reverse_pipe__(self, cnvt_range: bool=True, is_detach: bool=False, to_device: Union[str, torch.device]=None, to_unint8: bool=False, to_channel: str=None, to_np: bool=False, to_pil: bool=False):
        trans = [self.__forwar_reverse_fn__()[1]]
        detach_trans = [transforms.Lambda(lambda x: x.detach())]
        cpu_trans = [transforms.Lambda(lambda x: x.cpu())]
        unint8_trans = [transforms.Lambda(lambda x: x.mul_(255).add_(0.5).clamp_(0, 255).to(torch.uint8))]
        channel_last_trans = [transforms.Lambda(lambda x: self.__to_images_channel_last__(data=x))]
        channel_first_trans = [transforms.Lambda(lambda x: self.__to_images_channel_first__(data=x))]
        np_trans = [transforms.Lambda(lambda x: x.numpy())]
        pil_trans = [transforms.ToPILImage(mode='RGB')]
        
        if to_pil:
            return trans + detach_trans + cpu_trans + unint8_trans + channel_last_trans + np_trans + pil_trans
        
        res_trans = []
        if cnvt_range:
            res_trans = res_trans + trans
            
        if detach_trans:
            res_trans = res_trans + detach_trans
        
        if to_device is not None:
            res_trans = res_trans + [transforms.Lambda(lambda x: x.to(to_device))]
            
        if to_unint8:
            res_trans = res_trans + unint8_trans
            
        if to_channel == MultiDataset.CHANNEL_FIRST:
            res_trans = res_trans + channel_first_trans
        elif to_channel == MultiDataset.CHANNEL_LAST:
            res_trans = res_trans + channel_last_trans
            
        if to_np:
            res_trans = res_trans + detach_trans + cpu_trans + np_trans
        return res_trans
    
    def __check_image_chennel__(self, data: torch.Tensor, detected_channel_num: List[int]=[1, 3, 4]):
        if data.shape[-1] in detected_channel_num:
            # Chennel Last
            return MultiDataset.CHANNEL_LAST
        elif data.shape[-3] in detected_channel_num:
            # Chennel First
            return MultiDataset.CHANNEL_FIRST
        else:
            raise ValueError(f"Can only detect channel first or last")
        
    def __check_images_chennel__(self, data: Union[torch.Tensor, List[torch.Tensor]], detected_channel_num: List[int]=[1, 3, 4]):
        if isinstance(data, list):
            return [self.__check_image_chennel__(data=x, detected_channel_num=detected_channel_num) for x in data]
        elif isinstance(data, torch.Tensor):
            return self.__check_image_chennel__(data=data, detected_channel_num=detected_channel_num)
        else:
            raise TypeError(f"Can only accept torch.Tensor or List[torch.Tensor] not {type(data)}")
        
    def __image_channel__(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        return data.permute(1, 2, 0)
    
    def __force_image_channel_first__(self, data: torch.Tensor):
        dim: int = data.dim()
        perm: List[int] = [i for i in range(dim)]
        channel_dim = perm[-1]
        perm = perm[:-1]
        perm.insert(-2, channel_dim)
        return data.permute(*perm)
    
    def __force_images_channel_first__(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(data, list):
            return [self.__force_image_channel_first__(data=x) for x in data]
        elif isinstance(data, torch.Tensor):
            return self.__force_image_channel_first__(data=data)
        else:
            raise TypeError(f"Arguement data type, {type(data)},  is not supported, should be Union[torch.Tensor, List[torch.Tensor]]")
    
    def __to_image_channel_first__(self, data: torch.Tensor, is_force: bool=False):
        if self.__check_image_chennel__(data=data) == MultiDataset.CHANNEL_LAST or is_force:
            return self.__force_image_channel_first__(data=data)
        return data
    
    def __to_images_channel_first__(self, data: Union[torch.Tensor, List[torch.Tensor]], is_force: bool=False):
        if isinstance(data, list):
            return [self.__to_image_channel_first__(data=x, is_force=is_force) for x in data]
        elif isinstance(data, torch.Tensor):
            return self.__to_image_channel_first__(data=data, is_force=is_force)
        else:
            raise TypeError(f"Arguement data type, {type(data)},  is not supported, should be Union[torch.Tensor, List[torch.Tensor]]")
        
    def __force_image_channel_last__(self, data: torch.Tensor):
        dim: int = data.dim()
        perm: List[int] = [i for i in range(dim)]
        channel_dim = perm[-3]
        perm = perm[:-3] + perm[-2:]
        perm = perm + [channel_dim]
        # print(f"Channel Last perm: {perm}")
        return data.permute(*perm)
    
    def __force_images_channel_last__(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(data, list):
            return [self.__force_image_channel_last__(data=x) for x in data]
        elif isinstance(data, torch.Tensor):
            return self.__force_image_channel_last__(data=data)
        else:
            raise TypeError(f"Arguement data type, {type(data)},  is not supported, should be Union[torch.Tensor, List[torch.Tensor]]")
        
    def __to_image_channel_last__(self, data: torch.Tensor, is_force: bool=False):
        if self.__check_image_chennel__(data=data) == MultiDataset.CHANNEL_FIRST or is_force:
            # print("Execute to_channel_last")
            return self.__force_images_channel_last__(data=data)
        return data
    
    def __to_images_channel_last__(self, data: Union[torch.Tensor, List[torch.Tensor]], is_force: bool=False):
        if isinstance(data, list):
            return [self.__to_image_channel_last__(data=x, is_force=is_force) for x in data]
        elif isinstance(data, torch.Tensor):
            return self.__to_image_channel_last__(data=data, is_force=is_force)
        else:
            raise TypeError(f"Arguement data type, {type(data)},  is not supported, should be Union[torch.Tensor, List[torch.Tensor]]")
        
    def tensor2imgs(self, images: [torch.Tensor, List[torch.Tensor]], cnvt_range: bool=True, is_detach: bool=False, to_device: Union[str, torch.device]=None, to_unint8: bool=False, to_channel: str=None, to_np: bool=False, to_pil: bool=False):
        trans = transforms.Compose(self.__reverse_pipe__(cnvt_range=cnvt_range, is_detach=is_detach, to_device=to_device, to_unint8=to_unint8, to_channel=to_channel, to_np=to_np, to_pil=to_pil))
        if isinstance(images, list):
            return [trans(img) for img in images]
        else:
            return trans(images)
    
    def save_images(self, images: [torch.Tensor, List[torch.Tensor]], file_names: Union[str, List[str]]):
        if isinstance(images, list) and isinstance(file_names, list):
            if len(images) != len(file_names):
                raise ValueError(f"The arguement images and file_names should be lists with equal length, not {len(images)} and {len(file_names)}.")
        rev_images = self.tensor2imgs(images=images)
        if isinstance(rev_images, list):
            for i, r_img in enumerate(rev_images):
                utils.save_image(r_img, fp=file_names[i])
        else:
            utils.save_image(rev_images, fp=file_names)
            
class Optimizer:
    OPTIM_ADAM: str = "ADAM"
    
    @staticmethod
    def get_tainable_param(trainable_param: torch.Tensor):
        return torch.nn.Parameter(trainable_param).requires_grad_(requires_grad=True)
    
    @staticmethod
    def optim_generator(name: str, lr: float, **kwargs):
        if name == Optimizer.OPTIM_ADAM:
            opt = partial(torch.optim.Adam, lr=lr, **kwargs)
        else:
            raise ValueError(f"")
        return opt
    
class LossFn:
    METRIC_L1: str = "L1"
    METRIC_L2: str = "L2"
    # METRIC_L2_PER_ENTRY: str = "L2_PER_ENTRY"
    METRIC_FOURIER: str = "FOURIER"
    METRIC_SSIM: str = "SSIM"
    METRIC_LPIPS: str = "LPIPS"
    
    @staticmethod
    def mse(x0: torch.Tensor, pred_x0: torch.Tensor, reduction: str) -> torch.Tensor:
        return ((x0 - pred_x0) ** 2).sqrt().mean(list(range(x0.dim()))[1:])
    
    @staticmethod
    def get_loss_fn(metric: str, reduction: str='mean'):
        if metric == LossFn.METRIC_L1:
            if reduction == 'mean':
                criterion = lambda x, y: torch.nn.L1Loss(reduction='none')(x, y).mean(list(range(x.dim()))[1:])
            elif reduction == 'sum':
                criterion = lambda x, y: torch.nn.L1Loss(reduction='none')(x, y).sum(list(range(x.dim()))[1:])
            else:
                criterion = lambda x, y: torch.nn.L1Loss(reduction='none')(x, y)
        elif metric == LossFn.METRIC_L2:
            if reduction == 'mean':
                criterion = lambda x, y: torch.nn.MSELoss(reduction='none')(x, y).mean(list(range(x.dim()))[1:])
            elif reduction == 'sum':
                criterion = lambda x, y: torch.nn.MSELoss(reduction='none')(x, y).sum(list(range(x.dim()))[1:])
            else:
                criterion = lambda x, y: torch.nn.MSELoss(reduction='none')(x, y)
        # elif metric == LossFn.METRIC_SSIM:
        #     criterion = SSIMLoss()
        # elif metric == LossFn.METRIC_PSNR:
        #     criterion = psnr
        # elif metric == LossFn.METRIC_LPIPS:
        #     criterion = lpips_fn
        # elif metric == LossFn.METRIC_FOURIER:
        #     criterion = lpips_fn
        else:
            raise ValueError(f"Arguement metric doesn't support {metric}")
        return criterion

class ModelSched:
    MD_CLASS_CM: str = "MD_CM"
    
    MD_NAME_CD_L2_IMAGENET64: str = "CD_L2_IMAGENET64"
    MD_NAME_CD_LPIPS_IMAGENET64: str = "CD_LPIPS_IMAGENET64"
    MD_NAME_CT_IMAGENET64: str = "CT_IMAGENET64"
    
    MD_NAME_CD_L2_BEDROOM256: str = "CD_L2_BEDROOM256"
    MD_NAME_CD_LPIPS_BEDROOM256: str = "CD_LPIPS_BEDROOM256"
    MD_NAME_CT_BEDROOM256: str = "CT_BEDROOM256"
    
    MD_ID_CD_L2_IMAGENET64: str = "openai/diffusers-cd_imagenet64_l2"
    MD_ID_CD_LPIPS_IMAGENET64: str = "openai/diffusers-cd_imagenet64_lpips"
    MD_ID_CT_IMAGENET64: str = "openai/diffusers-ct_imagenet64"
    
    MD_ID_CD_L2_BEDROOM256: str = "openai/diffusers-cd_bedroom256_l2"
    MD_ID_CD_LPIPS_BEDROOM256: str = "openai/diffusers-cd_bedroom256_lpips"
    MD_ID_CT_BEDROOM256: str = "openai/diffusers-ct_bedroom256"
    
    @staticmethod
    def get_md_id(md_name: str):
        if md_name == ModelSched.MD_NAME_CD_L2_IMAGENET64:
            return ModelSched.MD_ID_CD_L2_IMAGENET64
        elif md_name == ModelSched.MD_NAME_CD_LPIPS_IMAGENET64:
            return ModelSched.MD_ID_CD_LPIPS_IMAGENET64
        elif md_name == ModelSched.MD_NAME_CT_IMAGENET64:
            return ModelSched.MD_ID_CT_IMAGENET64
        elif md_name == ModelSched.MD_NAME_CD_L2_BEDROOM256:
            return ModelSched.MD_ID_CD_L2_BEDROOM256
        elif md_name == ModelSched.MD_NAME_CD_LPIPS_BEDROOM256:
            return ModelSched.MD_ID_CD_LPIPS_BEDROOM256
        elif md_name == ModelSched.MD_NAME_CT_BEDROOM256:
            return ModelSched.MD_ID_CT_BEDROOM256
        else:
            raise ValueError(f"Arguement md_name, {md_name}, doesn't support ")
    
    @staticmethod
    def get_model_sched(model_type: str, model_id: str):
        vae = None
        # model, vae, noise_sched, get_pipeline, lora_layers = DiffuserModelSched.get_model_sched(ckpt=ep_model_path, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type, rank=config.lora_rank, alpha=config.lora_alpha)
        pipe: DiffusionPipeline = ConsistencyModelPipeline.from_pretrained(ModelSched.get_md_id(model_id), torch_dtype=torch.float16)
        return pipe, pipe.unet.eval(), vae, pipe.scheduler
    
    @staticmethod
    def all_compile(*args):
        comps: List = []
        for comp in args:
            if comp is None:
                comps.append(comp)
            else:
                comps.append(torch.compile(comp, mode="reduce-overhead", fullgraph=True))
        return comps

    @staticmethod
    def all_to_device(*args, device: Union[str, torch.device]):
        comps: List = []
        for comp in args:
            if hasattr(comp, 'to'):
                comps.append(comp.to(device))
            else:
                comps.append(comp)
        return comps
    
    

    # return (latents - epsilon) / 








@dataclass    
class Config:
    project: str = 'TestAdvTracer'
    name: str = None
    ds_root: str = "fake_images/CD_L2_IMAGENET64"
    repeat: int = 32
    vmin_out: int = -1
    vmax_out: int = 1
    model_type: str = ModelSched.MD_CLASS_CM
    model_id: str = ModelSched.MD_NAME_CD_L2_IMAGENET64
    loss_metric: str = LossFn.METRIC_L2
    optim_type: str = Optimizer.OPTIM_ADAM
    lr: float = 0.01
    num_warmup_steps: int = None
    num_training_steps: int = None
    num_cycles: int = 5
    num_classes: int = 1000
    sample_num: int = 3000
    path: str = os.path.join("fake_images", model_id)
    num_inference_steps: int = 1
    prompts: List[int] = None
    # prompts: List[int] = None
    prompts_arg_name: str = 'class_labels'
    batch_size: int = None
    device: Union[str, torch.device] = 'cuda:3'
    seed: int = 0
    max_iter: int = 1000
    grid_size: int = 16
    animate_duration: int = 10
    training_sample_folder: str = 'training_samples'
    
    def __naming_fn__(self):
        return f"res-{self.ds_root.replace('images/', '')}-{self.model_id}-{self.loss_metric}-lr{self.lr}-{self.loss_metric}-it{self.max_iter}-lrnc{self.num_cycles}-bs{self.batch_size}"
    
    def __post_init__(self):
        self.prompts = [i for i in range(self.num_classes)]
        # assert self.batch_size == None
        self.batch_size = self.repeat
        self.num_warmup_steps = int(self.max_iter * 0.1)
        self.num_training_steps = self.max_iter
        
        # Should be last one
        self.name = f"{self.__naming_fn__()}"
        
    def re_init__(self):
        self.__post_init__()
        
    # def make_dir(self):
    #     os.makedirs(self.name, exist_ok=True)
    #     os.makedirs(os.path.join(self.name, self.training_sample_folder), exist_ok=True)
    #     return self
    
    
    
        
    
        
    

    

        # return self.__loader__(data=src_ls)
        
    
    
        
        
    
    
    
    
        
    
        
    
        
    
            
class Optimizer:
    OPTIM_ADAM: str = "ADAM"
    
    @staticmethod
    
    @staticmethod
    
class LossFn:
    METRIC_L1: str = "L1"
    METRIC_L2: str = "L2"
    # METRIC_L2_PER_ENTRY: str = "L2_PER_ENTRY"
    METRIC_FOURIER: str = "FOURIER"
    METRIC_SSIM: str = "SSIM"
    METRIC_LPIPS: str = "LPIPS"
    
    @staticmethod
    
    @staticmethod

class ModelSched:
    MD_CLASS_CM: str = "MD_CM"
    
    MD_NAME_CD_L2_IMAGENET64: str = "CD_L2_IMAGENET64"
    MD_NAME_CD_LPIPS_IMAGENET64: str = "CD_LPIPS_IMAGENET64"
    MD_NAME_CT_IMAGENET64: str = "CT_IMAGENET64"
    
    MD_NAME_CD_L2_BEDROOM256: str = "CD_L2_BEDROOM256"
    MD_NAME_CD_LPIPS_BEDROOM256: str = "CD_LPIPS_BEDROOM256"
    MD_NAME_CT_BEDROOM256: str = "CT_BEDROOM256"
    
    MD_ID_CD_L2_IMAGENET64: str = "openai/diffusers-cd_imagenet64_l2"
    MD_ID_CD_LPIPS_IMAGENET64: str = "openai/diffusers-cd_imagenet64_lpips"
    MD_ID_CT_IMAGENET64: str = "openai/diffusers-ct_imagenet64"
    
    MD_ID_CD_L2_BEDROOM256: str = "openai/diffusers-cd_bedroom256_l2"
    MD_ID_CD_LPIPS_BEDROOM256: str = "openai/diffusers-cd_bedroom256_lpips"
    MD_ID_CT_BEDROOM256: str = "openai/diffusers-ct_bedroom256"
    
    @staticmethod
    
    @staticmethod
    
    @staticmethod

    @staticmethod
    
def auto_make_grids(samples: torch.Tensor):
    """
    Input/Output: Channel first
    """
    sample_grids = []
    for i in range(len(samples)):
        # print(f"Sample Grid shape: {Samples.auto_make_grid(samples[i]).shape}")
        sample_grids.append(auto_make_grid(samples[i]))
    sample_grids = torch.stack(sample_grids)
    return sample_grids
    
def auto_make_grid(sample: torch.Tensor, vmin_out: Union[float, int]=0, vmax_out: Union[float, int]=1):
    """
    Input/Output: Channel first
    """
    nrow = ceil(sqrt(len(sample)))
    return utils.make_grid(sample, nrow=nrow)

def single_step_denoise(pipeline, latents: torch.Tensor, from_t: int=0):
    return (pipeline.inference(batch_size=len(latents), num_inference_steps=1, class_labels=None, latents=latents, output_type='pt', is_pbar=False).images - 0.5) * 2
    # return (latents - epsilon) / 

def get_dataset(root: str, size: Union[int, Tuple[int], List[int]], repeat: int=1, vmin_out: int=-1, vmax_out: int=1):
    return MultiDataset(src=root, size=size, repeat=repeat, vmin_out=vmin_out, vmax_out=vmax_out)

def get_dataloader(dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def batchify(data, batch_size: int):
    res: list = []
    for i in range(0, len(data), batch_size):  
        res.append(data[i:i + batch_size])
    return res

def get_lr(optim, lr_sched=None):
    if optim is not None:
        return optim.param_groups[0]['lr']
    elif lr_sched is not None:
        return lr_sched.get_last_lr()[0]
    else:
        raise ValueError(f"Arguement optim and lr_sched should not be None in the same time")

def sampling(pipeline, num: int, path: str, num_inference_steps: int=1, prompts: List=None, prompts_arg_name: str=None, batch_size: int=32, generator: torch.Generator=None):
    if prompts is None or len(prompts) == 0:
        used_prompts: List = [None for i in range(num)]
    else:
        used_prompts: List = [prompts[i % len(prompts)] for i in range(num)]
        
    generator = set_generator(generator=generator)
    
    os.makedirs(path, exist_ok=True)
    for batch_idx, prompt in enumerate(batchify(data=used_prompts, batch_size=batch_size)):
        batch_size: int = len(prompt)
        if prompts is None:
            prompt = None
            
        if prompts_arg_name is not None:
            images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, **{prompts_arg_name: prompt}).images
        else:
            if isinstance(prompt, int):
                images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, class_labels=prompt).images
            elif isinstance(prompt, str):
                images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, prompt=prompt).images
            else:
                images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, prompt=prompt).images
            
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f"{batch_idx* batch_size + idx}.jpg"))

def tmp_ds():
    ds = get_dataset(root="fake_images/celeba_hq_256_ddpm", size=64)
    print(f"ds[0:5] Len: {len(ds[0:5])}, len(ds[0:5][0]): {len(ds[0:5][0])}")
    print(f"ds.to_tensor() Len: {len(ds.to_tensor())}, ds.to_tensor()[0].shape: {ds.to_tensor()[0].shape}, ds.to_tensor()[1].shape: {ds.to_tensor()[1].shape}, ds.to_tensor()[2].shape: {ds.to_tensor()[2].shape}")
    print(f"Max: {ds[0][0].max()}, Min: {ds[0][0].min()}")
    # utils.save_image(ds.tensor2imgs(ds[0]), fp='test.jpg')
    ds.save_images(ds[0][0], 'test.jpg')
    ds.save_images([ds[0:2][0][0], ds[0:2][1][0]], ['test.jpg', 'test1.jpg'])

def tmp_fn():
    model_type: str = ModelSched.MD_CLASS_CM
    model_id: str = ModelSched.MD_NAME_CD_LPIPS_IMAGENET64
    num_classes: int = 1000
    sample_num: int = 3000
    path: str = os.path.join("fake_images", model_id)
    num_inference_steps: int = 1
    prompts: List[int] = [i for i in range(num_classes)]
    # prompts: List[int] = None
    prompts_arg_name: str = 'class_labels'
    batch_size: int = 32
    device: Union[str, torch.device] = 'cuda:1'
    seed: int = 0
    
    g = torch.Generator(device=device).manual_seed(seed)
    
    pipe, unet, vae, scheduler = ModelSched.get_model_sched(model_type=model_type, model_id=model_id)
    pipe, unet, vae, scheduler = ModelSched.all_to_device(pipe, unet, vae, scheduler, device=device)
    pipe.unet, unet, vae = ModelSched.all_compile(pipe.unet, unet, vae)
    sampling(pipeline=pipe, num=sample_num, path=path, num_inference_steps=num_inference_steps, prompts=prompts, prompts_arg_name=prompts_arg_name, batch_size=batch_size, generator=g)

@dataclass    
class Config:
    project: str = 'TestAdvTracer'
    name: str = None
    ds_root: str = "fake_images/CD_L2_IMAGENET64"
    repeat: int = 32
    vmin_out: int = -1
    vmax_out: int = 1
    model_type: str = ModelSched.MD_CLASS_CM
    model_id: str = ModelSched.MD_NAME_CD_L2_IMAGENET64
    loss_metric: str = LossFn.METRIC_L2
    optim_type: str = Optimizer.OPTIM_ADAM
    lr: float = 0.01
    num_warmup_steps: int = None
    num_training_steps: int = None
    num_cycles: int = 5
    num_classes: int = 1000
    sample_num: int = 3000
    path: str = os.path.join("fake_images", model_id)
    num_inference_steps: int = 1
    prompts: List[int] = None
    # prompts: List[int] = None
    prompts_arg_name: str = 'class_labels'
    batch_size: int = None
    device: Union[str, torch.device] = 'cuda:3'
    seed: int = 0
    max_iter: int = 1000
    grid_size: int = 16
    animate_duration: int = 10
    training_sample_folder: str = 'training_samples'
    
    
        
        
    # def make_dir(self):
    #     os.makedirs(self.name, exist_ok=True)
    #     os.makedirs(os.path.join(self.name, self.training_sample_folder), exist_ok=True)
    #     return self
    
def Dataclass2Dict(data):
    data_dir: List[str] = dir(data)
    res: dict = {}
    for field in data_dir:
        if field[:2] != '__' and field[-2:] != '__':
            res[field] = getattr(data, field)
    return res
    
def parse_config():
    config: Config = Config()
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--project", "-pj", type=str, default=None, required=True, help="Name of the project")
    parser.add_argument("--ds_root", "-dsr", type=str, default=None, required=True, help="Root of dataset")
    parser.add_argument("--repeat", "-r", type=int, default=None, required=True, help="The repetition of data points")
    parser.add_argument("--model_type", "-mt", type=str, default=None, required=True, help="The detection model type")
    parser.add_argument("--model_id", "-mi", type=str, default=None, required=True, help="The detection model ID")
    parser.add_argument("--loss_metric", "-lm", type=str, default=None, required=True, help="The distance metric between ground-truth and predicted image")
    parser.add_argument("--lr", "-lr", type=float, default=None, required=True, help="The learning rate of optimizer")
    parser.add_argument("--max_iter", "-it", type=int, default=None, required=True, help="The maximum iteration number")
    parser.add_argument("--num_cycles", "-nc", type=int, default=None, required=True, help="The nnumber of learning rate scheduler cycle")
    parser.add_argument("--device", "-d", type=str, default=None, required=True, help="The computing unit that is used")
    
    args = parser.parse_args()
    
    config.project = args.project
    config.ds_root = args.ds_root
    config.repeat = args.repeat
    config.model_type = args.model_type
    config.model_id = args.model_id
    config.loss_metric = args.loss_metric
    config.lr = args.lr
    config.max_iter = args.max_iter
    config.num_cycles = args.num_cycles
    config.device = args.device
    
    config.re_init__()
    return config
    
def tmp_train():
    config: Config = parse_config()
    # config: Config = Config()
    accelerator = Accelerator(log_with=["wandb", LoggerType.TENSORBOARD])
    
    
    g = torch.Generator(device=config.device).manual_seed(config.seed)
    
    pipe, unet, vae, scheduler = ModelSched.get_model_sched(model_type=config.model_type, model_id=config.model_id)
    loss_fn = LossFn.get_loss_fn(metric=config.loss_metric)
    pipe, unet, vae, scheduler, loss_fn = ModelSched.all_to_device(pipe, unet, vae, scheduler, loss_fn, device=config.device)
    print(f"Compiling Models...")
    pipe.unet, unet, vae = ModelSched.all_compile(pipe.unet, unet, vae)
    print(f"Models Compiled")
    
    image_size: Union[int, List[int], Tuple[int]] = pipe.unet.config.sample_size
    
    ds = get_dataset(root=config.ds_root, size=image_size, repeat=config.repeat, vmin_out=config.vmin_out, vmax_out=config.vmax_out)
    dl = get_dataloader(dataset=ds, batch_size=config.batch_size)
    
    # wandb.init(project=config.project, name=config.name, id=config.name, settings=wandb.Settings(start_method="fork"))
    accelerator.init_trackers(project_name=config.project, config=config, init_kwargs={"wandb":{"name": config.name, "id": config.name, "settings": wandb.Settings(start_method="fork")}})
    os.makedirs(config.name, exist_ok=True)
    with open(os.path.join(config.name, "config.json"), "w") as f: 
        json.dump(Dataclass2Dict(data=config), f, indent=4)
    try:
        for batch_idx, batch in enumerate(dl):
            if batch_idx >= 200:
                break        
            
            img, noise, img_id = batch
            img, noise = ModelSched.all_to_device(img, noise, device=config.device)
            noise = Optimizer.get_tainable_param(noise)
            optim = Optimizer.optim_generator(name=config.optim_type, lr=config.lr)([noise])
            lr_sched = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optim, num_warmup_steps=config.num_warmup_steps, num_training_steps=config.num_training_steps, num_cycles=config.num_cycles)
            
            print(f"Batch {batch_idx}")
            
            # progress_bar = tqdm(total=max_iter, disable=not accelerator.is_local_main_process)
            progress_bar = tqdm(total=config.max_iter)
            progress_bar.set_description(f"Batch {batch_idx}")
            
            loss_log = []
            loss_vec_log = []
            lr_log = []
            step_log = []
            sample_log = []
            for step in tqdm(range(config.max_iter)):
                pred = single_step_denoise(pipeline=pipe, latents=noise.float(), from_t=0).float()
                loss_vec = loss_fn(img.float(), pred)
                loss = loss_vec.mean()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                if lr_sched is not None:
                    lr_sched.step()
                
                progress_bar.update(1)
                logs = {"AvgLoss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
                
                loss_log.append(loss.detach().cpu().item())
                loss_vec_log.append(loss_vec.detach().cpu())
                step_log.append(step)
                lr_log.append(get_lr(optim=optim, lr_sched=lr_sched))
                sample_log.append(ds.tensor2imgs(auto_make_grid(pred[:config.grid_size]), cnvt_range=False, is_detach=True, to_device='cpu'))
                # print(f"noise: {noise.shape}, min: {noise.min()}, max: {noise.max()}")
                
                torch.cuda.empty_cache()
                    
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(step_log, loss_log, label='Loss', color='b')
            ax[0].set_title("Loss V.S Step")
            ax[0].set_xlabel("Step")
            ax[0].legend()
            
            ax[1].plot(step_log, lr_log, label='LR', color='r')
            ax[1].set_title("LR V.S Step")
            ax[1].set_xlabel("Step")
            ax[1].legend()
            
            wandb_video = wandb.Video(np.stack(ds.tensor2imgs(sample_log, cnvt_range=True, to_unint8=True, to_np=True)), fps=config.max_iter // config.animate_duration)
            pil_ls = ds.tensor2imgs(sample_log, to_pil=True)
            accelerator.log({"loss": fig, 'sample_evol': wandb_video, 'final_loss': loss_log[-1], 'final_sample': wandb.Image(pil_ls[-1])}, step=batch_idx)
            
            
            
            work_dir = os.path.join(config.name, f'batch{batch_idx}')
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(os.path.join(work_dir, config.training_sample_folder), exist_ok=True)
            
            batch_config = Dataclass2Dict(data=config)
            batch_config['batch_idx'] = batch_idx
            with open(os.path.join(work_dir, "config.json"), "w") as f: 
                json.dump(batch_config, f, indent=4)
                
            for i, sample in enumerate(pil_ls):
                # print(f"sample: {sample.shape}, min: {sample.min()}, max: {sample.max()}")
                sample.save(os.path.join(work_dir, config.training_sample_folder, f"sample{i}.jpg"))
            pil_ls[-1].save(os.path.join(work_dir, f"final.jpg"))
            pil_ls[0].save(os.path.join(work_dir, f"animate.gif"), save_all=True, append_images=pil_ls[1:], duration=config.animate_duration, loop=0)
            save_file({'loss_vec_log': torch.stack(loss_vec_log), 'final_sample': pred.detach().cpu(), 'final_noise': noise.detach().cpu()}, os.path.join(work_dir, f"record.safetensors"))
            fig.savefig(os.path.join(work_dir, f"loss.jpg"))
            
            torch.cuda.empty_cache()
    finally:
        accelerator.end_training()
    
if __name__ == "__main__":
    # tmp_ds()
    # tmp_fn()
    tmp_train()
    
    