import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from typing import Callable, List, Optional, Union
from diffusers.utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
import PIL
from skimage import morphology
import torchvision.transforms.functional as F
import numpy as np
def tensor_mask_to_pil(tensor_mask, clean=False):
    # Normalize
    tensor_mask = (tensor_mask) / (tensor_mask.max())

    # Binarizing and returning the mask object
    tensor_mask = (tensor_mask>0.5)

    mask = tensor_mask.cpu().squeeze().numpy()
    if clean:
        mask = morphology.remove_small_objects(mask, min_size=16)

    pil_mask = Image.fromarray(mask).convert('RGB').resize((512, 512))
    return pil_mask

def tensor_mask_list_to_pil(mask_list_tesnor, clean=False, num_timesteps=50):
    retval = []
    for i in range(mask_list_tesnor.shape[0]):
        pil_mask = tensor_mask_to_pil(mask_list_tesnor[i], clean=clean)
        retval.append(pil_mask)
    empty_mask = mask_list_tesnor[0]*0
    empty_pil_mask = tensor_mask_to_pil(empty_mask, clean=clean)
    while num_timesteps > len(retval):
        retval.insert(0, empty_pil_mask)
    return retval

class ProgressiveMaskingInpaint(StableDiffusionInpaintPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        min_mask_idx: int = 0,
        max_mask_idx: int = 50,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInpaintPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

        >>> init_image = download_image(img_url).resize((512, 512))
        >>> mask_image = download_image(mask_url).resize((512, 512))

        >>> pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        if mask_image is None:
            raise ValueError("`mask_image` input cannot be undefined.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 8. Check that sizes of mask, masked image and latents match
        # num_channels_mask = mask.shape[1]
        # num_channels_masked_image = masked_image_latents.shape[1]
        # if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
        #     raise ValueError(
        #         f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
        #         f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
        #         f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
        #         f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
        #         " `pipeline.unet` or your `mask_image` or `image` input."
        #     )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        masks_list = mask_image
        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if min_mask_idx < num_inference_steps -i: 
                    mask_image_it = masks_list[i]
                else:
                    mask_image_it = masks_list[-min_mask_idx]
                if max_mask_idx < num_inference_steps -i:
                    mask_image_it = masks_list[0]
                # 4. Preprocess mask and image
                mask, masked_image = prepare_mask_and_masked_image(image, mask_image_it)
                # mask = mask.repeat(1, 3, 1, 1)
                
                # 7. Prepare mask latent variables
                mask, masked_image_latents = self.prepare_mask_latents(
                    mask,
                    masked_image,
                    batch_size * num_images_per_prompt,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 11. Post-processing
        image = self.decode_latents(latents)

        # 12. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 13. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return (image, has_nsfw_concept)
        # if not return_dict:
        #     return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

   


class ProgressiveMaskingInpaintv1_5(StableDiffusionImg2ImgPipeline):

    @torch.no_grad()
    def ddim_inversion(self, query, image, generator, num_timesteps):
        

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
        timesteps = reversed(self.scheduler.timesteps)
        # with torch.autocast(device_type=device.type, dtype=torch.float32):
        print("INVERSION")
        with self.progress_bar(total=num_timesteps - 1) as progress_bar:
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
            return latent_list[::-1]

    @torch.no_grad()
    def get_latents(
        self,
        image: Optional[Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        device = self._execution_device

        # Preprocess image
        image = self.image_processor.preprocess(image)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        # Prepare latent variables
        latents_list = []
        image = image.to(device=device, dtype=self.unet.dtype)
        latents = self.vae.encode(image).latent_dist.sample(generator)
        latents = self.vae.config.scaling_factor * latents

        # get noise
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=self.unet.dtype)

        # get latents list
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps - 1) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_latents = self.scheduler.add_noise(latents, noise, t)
                latents_list.append(noise_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                # image = self.decode_latents(noise_latents)
                # image = self.image_processor.postprocess(image, output_type='pil')
                # image[0].save(f'./nos/{i}.png')
        return latents_list

    @torch.no_grad()
    def __call__(
        self,
        query:Optional[str] = None,
        latents_set: Optional[List[torch.FloatTensor]] = None,
        mask_list:Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        min_mask_idx: int = 0,
        max_mask_idx: int = 50,
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
        # consist of [uncond query]
        

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        # Prepare latent variables and mask
        latents = latents_set[0]

        # if len(mask.shape) == 2:
        #     tensor_mask = torch.cat([mask.unsqueeze(0)] * 4).unsqueeze(0)
        # elif len(mask.shape) == 3:
        #     tensor_mask = torch.cat([mask] * 4).unsqueeze(0)
        
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        ratio = num_inference_steps//50
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if min_mask_idx < num_inference_steps -i: 
                    mask_image_it = mask_list[i//ratio]
                else:
                    mask_image_it = mask_list[-min_mask_idx]
                if mask_image_it.sum() == 0:
                    if i+1<len(latents_set):
                        latents = latents_set[i+1] 
                    else:
                        latents = latents
                    continue
                if max_mask_idx < num_inference_steps -i:
                    mask_image_it = mask_list[0]
                if len(mask_image_it.shape) == 2:
                    tensor_mask = torch.cat([mask_image_it.unsqueeze(0)] * 4).unsqueeze(0)
                elif len(mask_image_it.shape) == 3:
                    tensor_mask = torch.cat([mask_image_it] * 4).unsqueeze(0)

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
                    latents = (1-tensor_mask)*latents_set[i+1] + tensor_mask*latents_query
                else:
                    latents = latents_query
                # progress_bar.update
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

            image = self.decode_latents(latents)
            image = self.image_processor.postprocess(image, output_type='pil')

        return image