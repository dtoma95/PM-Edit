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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from torch.utils.data import Dataset, DataLoader
from ssim import ssim

from dataset import PIEBatchDataset
from diffusers import DDIMScheduler
from modules.diffedit_v2 import DiffEdit_v2

import json

directory_path = "/nas/users/tomislav/experiment1/flow_matching_translation/results/ddim_dataset_stats/ddim_stablediff_latents/"

json_path = "result_dict_rank_merged.json"
file = os.path.join(directory_path, json_path)
with open(file, 'r') as file:
    data_dict = json.load(file)

def save_testing_cis(x_list, y_list, image):
    merged = torch.cat(y_list, dim=0).float()
    merged = (merged * 0.5) + 0.5
    merged = merged.clamp(0.0, 1.0)
    utils.save_image(merged, "/ssd1/tomislav/stable_diffusion/testing/y_list.png", nrow=10)

    merged = torch.cat(x_list, dim=0).float()
    merged = (merged * 0.5) + 0.5
    merged = merged.clamp(0.0, 1.0)
    utils.save_image(merged, "/ssd1/tomislav/stable_diffusion/testing/x_list.png", nrow=10)

    merged = (image.float() * 0.5) + 0.5
    utils.save_image(merged.clamp(0.0, 1.0), "/ssd1/tomislav/stable_diffusion/testing/original.png", nrow=1)


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

def progressive_masking(diffusion, img, prompt, device, rank=0, guidance_scale=7.5, strength=1., num_inference_steps=50, batch_size=1, start_t=-1, end_t=0,lamba_treshold=1.0, args=None):
    if start_t > 0:
        strength = start_t
    do_classifier_free_guidance = guidance_scale > 1.0
    
    height = diffusion.unet.config.sample_size * diffusion.vae_scale_factor
    width = diffusion.unet.config.sample_size * diffusion.vae_scale_factor
    end_iteration = int(end_t*num_inference_steps)
    print("LOOOK HEREHRE RE", prompt)
    prompt_embeds = diffusion._encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )
    # Preprocess image
    image = diffusion.image_processor.preprocess(img)
    image = image.repeat(1, 1, 1, 1).to(device)
    # prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1, 1)
    # set timesteps
    diffusion.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = diffusion.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(1)

    # Prepare latent variables
   
    first = True
    x = None
    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * diffusion.scheduler.order

    extra_step_kwargs = diffusion.prepare_extra_step_kwargs(None, 0.0)
    #For redundant batching
    x_redundants = [None]*batch_size
    noise_redundants = [None]*batch_size

    latents = diffusion.prepare_latents(image, latent_timestep*0, 1, 1, prompt_embeds.dtype, device, None)
    x_list = []
    mask_list = [] # for tourbleshoot
    masks_list = [] # retval
    total_masking_steps = num_inference_steps-end_iteration
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            
            if i > total_masking_steps and len(masks_list)>0:
                masks_list.append(masks_list[-1])
                continue
            iteration_outputs_x = []
            iteration_outputs_y = []
            for rb in range(batch_size):
                #hocu da nappravim da radi batch size ali ne paralelrno!!!!!
                # expand the latents if we are doing classifier free guidance
                x = x_redundants[rb]
                noise = noise_redundants[rb]
                if noise is None: 
                    noise = torch.randn_like(latents)
                    noise_redundants[rb] = noise
                
                y_t = diffusion.scheduler.add_noise(latents, noise, t)
                y_t_latent_model_input = torch.cat([y_t] * 2) if do_classifier_free_guidance else y_t
                y_t_latent_model_input = diffusion.scheduler.scale_model_input(y_t_latent_model_input, t)

                if x is not None: 
                    y_t = mask_old*x + (1-mask_old)*y_t
                    # y_t = mask_downsized*x + (1-mask_downsized)*y_t
                if x is None: 
                    x = y_t.clone().detach()
                    x_redundants[rb] = x
                x_latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                x_latent_model_input = diffusion.scheduler.scale_model_input(x_latent_model_input, t)
                out_x_noise_pred = diffusion.unet(x_latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                out_y_noise_pred = diffusion.unet(y_t_latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = out_x_noise_pred.chunk(2)
                    out_x_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    noise_pred_uncond, noise_pred_text = out_y_noise_pred.chunk(2)
                    out_y_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if args.do_ssim >= 1:
                    iteration_outputs_x.append(out_x_noise_pred)
                    iteration_outputs_y.append(out_y_noise_pred)
                else:
                    out_x = diffusion.scheduler.step(out_x_noise_pred, t, x, **extra_step_kwargs)
                    out_y = diffusion.scheduler.step(out_y_noise_pred, t, y_t, **extra_step_kwargs)
                    iteration_outputs_x.append(out_x.pred_original_sample)
                    iteration_outputs_y.append(out_y.pred_original_sample)

            decode_y_all = torch.cat(iteration_outputs_y)
            decode_x_all = torch.cat(iteration_outputs_x)

            # L1
            error = torch.abs(decode_y_all - decode_x_all)#.mean(dim=1, keepdim=True)
            error = error.mean(dim=0, keepdim=True)
            
            if first:
                mask_old = torch.zeros_like(latents)
                mask = mask_old
                first=False
                threshold=0
            else:
                threshold =  data_dict["l1_stats"]["mean"][::-1][i] + data_dict["l1_stats"]["std"][::-1][i]*lamba_treshold
                mask = (error.mean(dim=1, keepdim=True) > threshold).float().repeat(1, latents.shape[1], 1, 1)
                if args.do_ssim >= 1:
                    threshold =  data_dict["ssim_stats"]["mean"][::-1][i] + data_dict["ssim_stats"]["std"][::-1][i]*lamba_treshold
                    mask = (error.mean(dim=1, keepdim=True) > threshold).float().repeat(1, latents.shape[1], 1, 1)
            
            mask = 1 - (1-mask_old)*(1-mask)
            mask = mask.to(x.dtype)
            mask_old = mask
            
            for rb in range(batch_size):
                x = x_redundants[rb]

                noise = noise_redundants[rb]
                y_t = diffusion.scheduler.add_noise(latents, noise, t)

                x = mask_old*x + (1-mask_old)*y_t
                latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else latents_t
                latent_model_input = diffusion.scheduler.scale_model_input(latent_model_input, t)

                out_x_noise_pred = diffusion.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
    
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = out_x_noise_pred.chunk(2)
                    out_x_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                out_x = diffusion.scheduler.step(out_x_noise_pred, t, x, **extra_step_kwargs)
                x_redundants[rb] = out_x.prev_sample

                if rb == 0 and rank == 0 and False:
                    decode_x = 1 / diffusion.vae.config.scaling_factor * out_x.pred_original_sample #.prev_sample
                    x_list.append(diffusion.vae.decode(decode_x).sample.detach().cpu())
                    mask_list.append(mask_old.detach().cpu().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)*2-1)
                # out = diffusion.ddim_sample(model, y_t, t, model_kwargs={}, eta=0.0)
                # x = out["sample"]
            if rank == 0:
                print(i, t, mask_old.mean(), threshold, lamba_treshold, mask_old.shape)
            if mask.sum() >0:
                masks_list.append(mask.mean(dim=1, keepdim=True).detach().cpu())
            
    if rank == 0 and False:      
        save_testing_cis(x_list, mask_list, image)
    print('', end="\r")
    return mask.mean(dim=1, keepdim=True), masks_list

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

    ddim_scheduler =  DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        subfolder="scheduler"
    )
    diffusion = DiffEdit_v2.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        safety_checker=None,
        torch_dtype=torch.float16,
        scheduler=ddim_scheduler,
    ).to(args.device)

    transform = None
    dataset = PIEBatchDataset(root_dir=args.data_path, transform=transform)
    override=False
    if args.num_samples == -1:
        override=True
        args.num_samples = len(dataset) #+ world_size
        print("SETTING TOTAL NUMBER OF SAMPLES TO:", args.num_samples)
    count = 0
    total = args.num_samples//world_size* 500

    
    loader = prepare_dataloader(dataset, args.batch_size)
    data_iter = iter(loader)
    epoch = 0
    while count * args.batch_size < total:
        file_count = count_files_os_walk(args.save_path_masks)
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
            img_file_path = os.path.join(args.save_path_images, "{img_filename}.pt".format(img_filename=dict_batch['filename'][j], epoch=0))
            if img_file_path == '/ssd1/tomislav/stable_diffusion/masksl=1.2s=1ss=1/mask_lists/7_change_attribute_material_40/1_artificial/1_animal/711000000001.jpg.pt':
                for cringe in range(1000):
                    print("JISU")
            if os.path.exists(img_file_path):
                count = count+1
                print("Iiteration", count * args.batch_size, "SKIPINNG", dict_batch['filename'][j], "FILE COUNT", file_count, "of", args.num_samples)
                skip=True
                
        if skip:
            continue


        print(dict_batch)
        img_name = os.path.join(dataset.root_dir, dict_batch['filename'][0]) #dict_batch["filename"]
        img = Image.open(img_name).convert('RGB').resize((512, 512))
        # prompt = dict_batch["original_prompt"] 
        prompt = dict_batch["editing_prompt"]
        print(prompt)


        mask, masks_list = progressive_masking(diffusion, img, prompt, args.device, args=args,
                                 rank=rank, batch_size=args.redundant_batches, end_t=args.end_t, start_t=args.start_t, lamba_treshold=args.lamba_treshold)
        img_filename = dict_batch['filename']
        for j in range(len(img_filename)):
            

            path_result = os.path.join(args.save_path_images, "{img_filename}.pt".format(img_filename=img_filename[j], epoch=0))
            os.makedirs(os.path.dirname(path_result), exist_ok=True)
            torch.save(torch.cat(masks_list, dim=0).to(torch.uint8), path_result)

            path_result = os.path.join(args.save_path_masks, "{img_filename}.png".format(img_filename=img_filename[j], epoch=0))
            os.makedirs(os.path.dirname(path_result), exist_ok=True)
            utils.save_image(mask[j].clamp(0.0, 1.0), path_result, nrow=1)
        count = count+1
        print("Iiteration", count * args.batch_size, "of", total, "FILE COUNT", file_count)
        
    print("sampling complete")

def save_dict(result_dict, save_path, rank):
    serializable_dict = {"model_path": result_dict["model_path"], 
            "data_path": result_dict["data_path"], 
            "total_samples": result_dict["total_samples"], 
            "l1_stats": {}, 
            "ssim_stats": {}}
    for key in result_dict["l1_stats"].keys():
        serializable_dict["l1_stats"][key] = result_dict["l1_stats"][key].tolist()
        serializable_dict["ssim_stats"][key] = result_dict["ssim_stats"][key].tolist()

    with open(os.path.join(save_path,f"result_dict_rank_{rank}.json"), "w") as json_file:
        json.dump(serializable_dict, json_file, indent=4)

def merge_stats(stats_1, stats_2, ratio):
    for key in stats_1.keys():
        stats_1[key] = stats_1[key]*ratio + stats_2[key]*(1-ratio)
    return stats_1


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        default='./ckpt/runwayml_sd_v1_5',
        type=str,
        help='path of the weight of stable diffusioninpaint pipeline '
    )
    parser.add_argument(
        '--data_path',
        default='/ssd1/dataset/PIE-Bench',
        type=str
    )
    parser.add_argument(
        '--save_path',
        default='/ssd1/tomislav/stable_diffusion/masks1.2',
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
    parser.add_argument("--lamba_treshold", type=float,
        dest="lamba_treshold", default=1.2,
        help="lamba_treshold")
    parser.add_argument("--redundant_batches", type=int,
        dest="redundant_batches", default=10,
        help="redundant_batches")
    parser.add_argument("--start_t", type=float,
        dest="start_t", default=1.0,
        help="start_t")
    parser.add_argument("--end_t", type=float,
        dest="end_t", default=0.0,
        help="end_t")
        
    parser.add_argument("--do_ssim", type=int,
        dest="do_ssim", default=1,
        help="do_ssim")
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
    args.save_path_images = os.path.join(args.save_path, "mask_lists")
    if not os.path.exists(args.save_path_images):
        os.mkdir(args.save_path_images)
    args.save_path_masks = os.path.join(args.save_path, "masks")
    if not os.path.exists(args.save_path_masks):
        os.mkdir(args.save_path_masks)
    with open(args.save_path+"/config.json", 'w') as f:
        f.write(json.dumps(vars(args),
            indent=4
        ))
    f.close()
    try:
        print("Running parallel scripts.")
        print(directory_path)
        mp.spawn(thread_main, args=(world_size, args), nprocs=world_size)
    except BaseException as e:
        raise e
        print(e)

        


