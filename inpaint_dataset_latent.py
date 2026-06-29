# sudo env PATH="$PATH" python main_parallel.py --output_dir='/ssd1/tomislav/flow_matching'

import sys

import argparse
import os
from PIL import Image
# import blobfile as bf
import numpy as np

from torchvision import datasets, transforms, utils

import torch

import datetime
import torch.nn as nn
import torch.nn.functional as F

import os
import json

import argparse
import sys
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from torch.utils.data import Dataset, DataLoader
from ssim import ssim

from dataset import PIEBatchDataset
from modules.diffedit_v2 import DiffEdit_v2
from diffusers import DDIMScheduler,StableDiffusionInpaintPipeline
from modules.pm_inpaint import ProgressiveMaskingInpaint, tensor_mask_list_to_pil, tensor_mask_to_pil, ProgressiveMaskingInpaintv1_5

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        # num_workers=0,
        drop_last=True
    )

def count_files_os_walk(directory_path):
    count = 0
    # os.walk yields a tuple of (root, dirs, files) for each directory
    for root, dirs, files in os.walk(directory_path):
        count += len(files)
    return count

def thread_main(rank, world_size, args):
    ddp_setup(rank, world_size)
    args.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'mps')

    print("creating model...")
    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )
    # model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    # model.to(args.device)
    # if args.use_fp16:
    #     model.convert_to_fp16()
    # model.eval()
    DDIM =  DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        subfolder="scheduler"
    )    
    if "runwayml_sd_v1_5" in args.ckpt_dir:
        pipe = ProgressiveMaskingInpaintv1_5.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_dir,
            safety_checker=None,
            torch_dtype=torch.float16,
            scheduler=DDIM,
        ).to(args.device)
    elif args.mask_list_inpaint == 0:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_dir,
            safety_checker=None,
            torch_dtype=torch.float16,
            scheduler=DDIM,
        ).to(args.device)
    else:
        pipe = ProgressiveMaskingInpaint.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_dir,
            safety_checker=None,
            torch_dtype=torch.float16,
            scheduler=DDIM,
        ).to(args.device)
        
    transform = None
    dataset = PIEBatchDataset(root_dir=args.data_path, transform=transform)
    
    if args.num_samples == -1:
        override=True
        args.num_samples = len(dataset) #+ world_size
        print("SETTING TOTAL NUMBER OF SAMPLES TO:", args.num_samples)
    count = 0
    total = args.num_samples//world_size *500

    
    loader = prepare_dataloader(dataset, args.batch_size)
    data_iter = iter(loader)
    epoch = 0

    do_clean = args.clean > 0
    while count * args.batch_size < total:
        file_count = count_files_os_walk(args.save_path_images)
        if file_count==args.num_samples:
            break
        try:
            dict_batch = next(data_iter) 
        except StopIteration:
            try:
                loader.sampler.set_epoch(epoch)
            except:
                pass
            epoch += 1
            data_iter = iter(loader)
            dict_batch = next(data_iter) 
        skip = False
        for j in range(len(dict_batch["filename"])):
            img_file_path = os.path.join(args.save_path_images, "{img_filename}.png".format(img_filename=dict_batch["filename"][j], epoch=0))
            if os.path.exists(img_file_path):
                count = count+1
                print("Iiteration", count * args.batch_size, "SKIPINNG", dict_batch["filename"][j], "FILE COUNT", file_count, "of", args.num_samples)
                skip=True
                
        if skip:
            continue


        print(dict_batch)
        img_name = os.path.join(dataset.root_dir, dict_batch['filename'][0]) #dict_batch["filename"]
        image = Image.open(img_name).convert('RGB').resize((512, 512))
        # prompt = dict_batch["original_prompt"] 
        prompt = dict_batch["editing_prompt"]
        print(prompt)
        # save memory and inference fast
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        # pipe.vae.enable_tiling()
        # pipe.enable_model_cpu_offload()

        
        if "runwayml_sd_v1_5" in args.ckpt_dir:
            if args.mask_list_inpaint == 0:
                mask_path = os.path.join(args.masks_path, "masks",  dict_batch['filename'][0]+".png")
                mask = Image.open(mask_path) 
                mask = torch.from_numpy(np.asarray(mask))[:, :, 0] 
                mask_list = mask.unsqueeze(0).unsqueeze(0)/255
                mask_list = mask_list.repeat(int(args.steps*args.start_t), 1, 1, 1)
                padding_size = 50 - mask_list.shape[0]
                mask_list = F.pad(mask_list, (0, 0, 0, 0, 0, 0, padding_size, 0)).to(args.device).half()
                print(mask_list.shape, "MASKLIST=0", mask_list.mean())
            else:
                mask_path = os.path.join(args.masks_path, "mask_lists", dict_batch['filename'][0]+".pt")
                mask_list = torch.load(mask_path).to(args.device)
                padding_size = 50 - mask_list.shape[0]
                mask_list = F.pad(mask_list, (0, 0, 0, 0, 0, 0, padding_size, 0)).to(args.device)
                print(mask_list.shape, "MASKLIST=1")
            # latents_set = pipe.get_latents(
            #     image=image,
            #     strength=args.strength,
            #     num_inference_steps=args.steps,
            #     generator=torch.Generator(device=args.device).manual_seed(args.seed)
            # )
            latents_inv_set = pipe.ddim_inversion(dict_batch["original_prompt"], 
                image, 
                generator=torch.Generator(device=args.device).manual_seed(args.seed), 
                num_timesteps=args.steps)
            result = pipe(
                query=dict_batch["editing_prompt"],
                latents_set=latents_inv_set,
                mask_list=mask_list,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                min_mask_idx=args.min_mask_idx,
                max_mask_idx=args.max_mask_idx,
                generator=torch.Generator(device=args.device).manual_seed(args.seed),
            )[0]
        elif args.mask_list_inpaint == 0:
            mask_path = os.path.join(args.masks_path, "masks",  dict_batch['filename'][0]+".png")
            mask = Image.open(mask_path)
            mask = torch.from_numpy(np.asarray(mask))[:, :, 0]
            mask = tensor_mask_to_pil(mask, clean=do_clean)
            # mask = Image.open(mask_path).convert('RGB').resize((512, 512))
            result = pipe(
                prompt=dict_batch["editing_prompt"],
                image=image,
                mask_image=mask,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                generator=torch.Generator(device=args.device).manual_seed(args.seed)
            ).images[0]
        else:
            mask_path = os.path.join(args.masks_path, "mask_lists", dict_batch['filename'][0]+".pt")
            mask_list = torch.load(mask_path)
            
            mask_list = tensor_mask_list_to_pil(mask_list, clean=do_clean)
            result = pipe(
                prompt=dict_batch["editing_prompt"],
                image=image,
                mask_image=mask_list,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                min_mask_idx=args.min_mask_idx,
                max_mask_idx=args.max_mask_idx,
                generator=torch.Generator(device=args.device).manual_seed(args.seed)
            )[0][0]
        img_filename = dict_batch['filename']
        for j in range(len(img_filename)):
            path_result = os.path.join(args.save_path_images, "{img_filename}.png".format(img_filename=img_filename[j], epoch=0))
            os.makedirs(os.path.dirname(path_result), exist_ok=True)
            result.save(path_result)

            # mask.save(path_result+"mask.png")
        count = count+1
        print("Iiteration", count * args.batch_size, "of", total, "FILE COUNT", file_count)
        
    print("sampling complete")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        default='./ckpt/runwayml_sd_v1_5',#runwayml_sd_v1_5 # './ckpt/inpainting-v1-2'
        type=str,
        help='path of the weight of stable diffusioninpaint pipeline '
    )
    parser.add_argument(
        '--data_path',
        default='/ssd1/dataset/PIE-Bench',
        type=str
    )
    parser.add_argument(
        '--masks_path',
        default='/nas/users/tomislav/experiment1/DiffEdit-PM-by-Stable-Diffusion/DiffeditMasks',
        type=str,
        help='path of the mask'
    )
    parser.add_argument(
        '--save_path',
        default='/ssd1/tomislav/stable_diffusion/diffedit_test',
        type=str
    )
    parser.add_argument(
        '--num_timesteps',
        default=50,
        type=int,
    )

    parser.add_argument(
        '--num_samples',
        default=-1,
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--steps',
        default=50,
        type=int,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--min_mask_idx',
        default=0,
        type=int,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--max_mask_idx',
        default=50,
        type=int,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--seed',
        default=2625,
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--scale',
        default=7.5,
        type=float,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--mask_list_inpaint',
        default=1,
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--clean',
        default=1,
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--strength',
        default=1.0,
        type=float,
        help='strength'
    )
    parser.add_argument(
        '--start_t',
        default=1.0,
        type=float,
        help='strength'
    )
    return parser

if __name__ == '__main__':

    args = create_argparser().parse_args()

    # ======================================================================
    # random seed
    # ======================================================================
    # pl.seed_everything(23141)
    world_size = torch.cuda.device_count()
    args.world_size = world_size

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path_images = os.path.join(args.save_path, "images")
    if not os.path.exists(args.save_path_images):
        os.mkdir(args.save_path_images)
    with open(args.save_path+"/config.json", 'w') as f:
        f.write(json.dumps(vars(args),
            indent=4
        ))
    f.close()
    try:
        print("Running parallel scripts.")
        # print(directory_path)
        mp.spawn(thread_main, args=(world_size, args), nprocs=world_size)
    except BaseException as e:
        raise e
        print(e)

        


