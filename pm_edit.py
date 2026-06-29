import torch
from PIL import Image
from diffusers import DDIMScheduler
import os
import argparse
import json
from modules.pm_edit import StableDiffusionProgressiveMasking 
import torchvision

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        default='./ckpt/runwayml_sd_v1_5',
        type=str,
        help='path of the weight of stable diffusioninpaint pipeline '
    )
    parser.add_argument(
        '--dt_stats_json',
        default='/nas/users/tomislav/experiment1/flow_matching_translation/results/ddim_dataset_stats/ddim_stablediff_latents/result_dict_rank_merged.json',
        type=str,
        help='path of the weight of stable diffusioninpaint pipeline '
    )
    
    parser.add_argument(
        '--image_dir',
        default=None,
        type=str,
        help='path of the image to be edited'
    )
    parser.add_argument(
        '--num_timesteps_mask',
        default=50,
        type=int,
        help='must be the same as the number of steps in dt_stats'
    )
    parser.add_argument(
        '--num_timesteps_inpaint',
        default=50,
        type=int,
        help='number of steps to take during inpainting, does not affect mask generation'
    )

    parser.add_argument(
        '--mask_batches',
        default=10,
        type=int,
    )
    parser.add_argument(
        '--reference',
        default=None,
        type=str,
        help='reference prompt'
    )
    parser.add_argument(
        '--query',
        default=None,
        type=str,
        help='edit prompt'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        help='path of the result to be saved'
    )
    parser.add_argument(
        '--lamba_treshold',
        default=1.0,
        type=float,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--scale',
        default=7.5,
        type=float,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--start_t',
        default=1.0,
        type=float,
        help='strength'
    )
    parser.add_argument(
        '--end_t',
        default=1.0,
        type=float,
        help='strength'
    )
    parser.add_argument(
        '--seed',
        default=10,
        type=int,
    )
    
    args = parser.parse_args()
    return args

args = get_args()

with open(args.dt_stats_json, 'r') as file:
    dt_stats = json.load(file)


init_image = Image.open(args.image_dir).convert('RGB').resize((512, 512))

device = torch.device('cuda')

DDIM =  DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        subfolder="scheduler"
    )  

pipeline = StableDiffusionProgressiveMasking.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_dir,
            safety_checker=None,
            torch_dtype=torch.float16,
            scheduler=DDIM,
        ).to(device)




pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_attention_slicing()
# pipeline.vae.enable_tiling()
# pipeline.enable_model_cpu_offload()

import time
with torch.no_grad():
    start = time.time()
    masks_list = pipeline.progressive_masking(image=init_image, 
                prompt=args.reference, 
                batch_size=args.mask_batches,
                start_t=args.start_t,
                end_t=args.end_t,
                lamba_treshold=args.lamba_treshold,
                dt_stats_dict=dt_stats)
    end_mask = time.time()
    print("MASK DONE", (end_mask-start))
    start_inpaint = time.time()
    latents_inv_set = pipeline.ddim_inversion(args.reference, 
                init_image, 
                strength=args.start_t,
                generator=torch.Generator(device=device).manual_seed(args.seed), 
                num_timesteps=args.num_timesteps_inpaint)
    print("LATENTS DONE")
    image_result = pipeline.inpaint_with_mask_list(
                query=args.query,
                latents_set=latents_inv_set,
                mask_list=masks_list,
                strength=args.start_t,
                num_inference_steps=args.num_timesteps_inpaint,
                guidance_scale=args.scale,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            )[0]
    end = time.time()
    print("INPAINTING DONE", (end-start_inpaint))
    print("Total time", (end-start))
final_mask = masks_list[-1].cpu().numpy()
print(type(final_mask), type(image_result))
print(final_mask.size)
pil_mask_unresized = Image.fromarray((final_mask.squeeze()*255).astype("uint8")).convert('RGB')
pil_mask_resized = pil_mask_unresized.resize((512, 512))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
image_result.save(f'{args.output_dir}/result_image.png')
pil_mask_unresized.save(f'{args.output_dir}/unresized_mask.png')
pil_mask_resized.save(f'{args.output_dir}/resized_mask.png')
masks_list = masks_list.float()
# merged = (merged * 0.5) + 0.5
masks_list = masks_list.clamp(0.0, 1.0)
torchvision.utils.save_image(masks_list, f'{args.output_dir}/masks_list.png', nrow=10)