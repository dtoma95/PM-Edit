import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from typing import Callable, List, Optional, Union
from diffusers.utils import randn_tensor
import PIL
from skimage import morphology
import torchvision.transforms.functional as F
import numpy as np

class StableDiffusionProgressiveMasking(StableDiffusionImg2ImgPipeline):

    @torch.no_grad()
    def progressive_masking(self, 
            image, 
            prompt, 
            guidance_scale=7.5, 
            num_inference_steps=50, 
            batch_size=1, 
            start_t=1.0, 
            end_t=0.0,
            lamba_treshold=1.0, 
            dt_stats_dict=None,
            do_noise_pred_discrepancy=True):
        device = self._execution_device
        
        skipped_steps_num = num_inference_steps - int(num_inference_steps*start_t)
        # skipped_steps_num = 0
        # print("SKIPS", skipped_steps_num)
        if start_t > 0:
            strength = start_t
        do_classifier_free_guidance = guidance_scale > 1.0
        
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor
        end_iteration = int(end_t*num_inference_steps)

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        # Preprocess image
        image = self.image_processor.preprocess(image)
        image = image.repeat(1, 1, 1, 1).to(device)
        # prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1, 1)
        # set timesteps
        num_inference_steps_old = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(1)

        # Prepare latent variables
        first = True
        x = None
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        extra_step_kwargs = self.prepare_extra_step_kwargs(None, 0.0)
        #For redundant batching
        x_redundants = [None]*batch_size
        noise_redundants = [None]*batch_size

        latents = self.prepare_latents(image, latent_timestep*0, 1, 1, prompt_embeds.dtype, device, None)
        x_list = []
        mask_list = [] # for tourbleshoot
        masks_list = [] # retval
        total_masking_steps = int(num_inference_steps_old*start_t)-int(num_inference_steps_old*end_t)
        with torch.no_grad():
            with self.progress_bar(total=total_masking_steps) as progress_bar:
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
                        
                        y_t = self.scheduler.add_noise(latents, noise, t)
                        y_t_latent_model_input = torch.cat([y_t] * 2) if do_classifier_free_guidance else y_t
                        y_t_latent_model_input = self.scheduler.scale_model_input(y_t_latent_model_input, t)

                        if x is not None: 
                            y_t = mask_old*x + (1-mask_old)*y_t
                            # y_t = mask_downsized*x + (1-mask_downsized)*y_t
                        if x is None: 
                            x = y_t.clone().detach()
                            x_redundants[rb] = x
                        x_latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                        x_latent_model_input = self.scheduler.scale_model_input(x_latent_model_input, t)
                        out_x_noise_pred = self.unet(x_latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                        out_y_noise_pred = self.unet(y_t_latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = out_x_noise_pred.chunk(2)
                            out_x_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            noise_pred_uncond, noise_pred_text = out_y_noise_pred.chunk(2)
                            out_y_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if do_noise_pred_discrepancy:
                            iteration_outputs_x.append(out_x_noise_pred)
                            iteration_outputs_y.append(out_y_noise_pred)
                        else:
                            out_x = self.scheduler.step(out_x_noise_pred, t, x, **extra_step_kwargs)
                            out_y = self.scheduler.step(out_y_noise_pred, t, y_t, **extra_step_kwargs)
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
                        
                        if do_noise_pred_discrepancy:
                            threshold =  dt_stats_dict["ssim_stats"]["mean"][::-1][skipped_steps_num+i] + dt_stats_dict["ssim_stats"]["std"][::-1][i]*lamba_treshold
                            mask = (error.mean(dim=1, keepdim=True) > threshold).float().repeat(1, latents.shape[1], 1, 1)
                        else:
                            threshold =  dt_stats_dict["l1_stats"]["mean"][::-1][skipped_steps_num+i] + dt_stats_dict["l1_stats"]["std"][::-1][i]*lamba_treshold
                        mask = (error.mean(dim=1, keepdim=True) > threshold).float().repeat(1, latents.shape[1], 1, 1)
                    mask = 1 - (1-mask_old)*(1-mask)
                    mask = mask.to(x.dtype)
                    mask_old = mask
                    
                    for rb in range(batch_size):
                        x = x_redundants[rb]
                        noise = noise_redundants[rb]
                        y_t = self.scheduler.add_noise(latents, noise, t)

                        x = mask_old*x + (1-mask_old)*y_t
                        latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else latents_t
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        out_x_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = out_x_noise_pred.chunk(2)
                            out_x_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        out_x = self.scheduler.step(out_x_noise_pred, t, x, **extra_step_kwargs)
                        x_redundants[rb] = out_x.prev_sample

                    masks_list.append(mask.mean(dim=1, keepdim=True).detach())
                    progress_bar.update()
        return torch.cat(masks_list, dim=0).to(device)

    @torch.no_grad()
    def ddim_inversion(self, query, image, generator, num_timesteps, strength):
        device = self._execution_device
        # Preprocess image
        image = self.image_processor.preprocess(image)

        image = image.to(device=device, dtype=self.unet.dtype)
        latent = self.vae.encode(image).latent_dist.sample(generator)
        latent = self.vae.config.scaling_factor * latent

        cond = self._encode_prompt(
            query,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        self.scheduler.set_timesteps(num_timesteps, device=device)
        latent_list=[latent]
        timesteps, num_inference_steps_short = self.get_timesteps(num_timesteps, strength, device)
        timesteps = reversed(timesteps)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
                progress_bar.update()
        return latent_list[::-1]

    @torch.no_grad()
    def inpaint_with_mask_list(
        self,
        query:Optional[str] = None,
        latents_set: Optional[List[torch.FloatTensor]] = None,
        mask_list:Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # get hyperparamemter
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # get embedding from reference prompt and query prompt, merge them
        query_prompt_embeds = self._encode_prompt(
            query,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        ratio = num_inference_steps//mask_list.shape[0]
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps_short = self.get_timesteps(num_inference_steps, strength, device)

        # Prepare latent variables and mask
        latents = latents_set[0]

        if latents.shape[1] != mask_list.shape[1]:
            mask_list = mask_list.repeat(1,latents.shape[1],1,1)
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps_short * self.scheduler.order

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                mask_image_it = mask_list[i//ratio].unsqueeze(0)

                if mask_image_it.sum() == 0:
                    if i+1<len(latents_set):
                        latents = latents_set[i+1] 
                    else:
                        latents = latents

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=query_prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond,noise_pred_query = noise_pred.chunk(2)
                    query_noise_residual = noise_pred_uncond + guidance_scale * (noise_pred_query - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_query = self.scheduler.step(query_noise_residual, t, latents, **extra_step_kwargs).prev_sample

                if i+1<len(latents_set):
                    latents = (1-mask_image_it)*latents_set[i+1] + mask_image_it*latents_query
                else:
                    latents = latents_query
                # progress_bar.update
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

            image = self.decode_latents(latents)
            image = self.image_processor.postprocess(image, output_type='pil')

        return image

    @torch.no_grad()
    def __call__(
        self,
        init_image,
        dt_stats,
        reference: Optional[str] = None,
        query: Optional[str] = None,

        lamba_treshold: Optional[float] = 1.4,
        mask_batches: Optional[int] = 50,
        start_t: Optional[float] = 0.8,
        end_t: Optional[float] = 0.5,
        num_inference_steps_mask: Optional[int] = 50,
        num_inference_steps_inpaint: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = 2112.,
    ):
        masks_list = pipeline.progressive_masking(image=init_image, 
                source_prompt=reference, 
                batch_size=mask_batches,
                start_t=start_t,
                end_t=end_t,
                lamba_treshold=lamba_treshold,
                dt_stats_dict=dt_stats,
                num_inference_steps=num_inference_steps_mask,
                )

        latents_inv_set = pipeline.ddim_inversion(reference, 
                image, 
                generator=torch.Generator(device="cuda").manual_seed(seed), 
                num_timesteps=num_inference_steps_inpaint)

        result = pipeline.inpaint_with_mask_list(
                query=query,
                latents_set=latents_inv_set,
                mask_list=mask_list,
                strength=start_t,
                num_inference_steps=num_inference_steps_inpaint,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )[0]
        return result