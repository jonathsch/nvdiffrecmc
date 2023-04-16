from typing import Optional

import torch
from torch import nn
import torch.functional as F
from lightning.fabric import Fabric
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


class StableDiffusionSDS(nn.Module):
    """
    Score Distillation Sampling with Stable Diffusion (https://arxiv.org/abs/2211.07600)

    Adapted from https://github.com/eladrich/latent-nerf/blob/main/src/stable_diffusion.py

    Args:
        model_name (str): Name of the model to use for the VAE and UNet.
        latent_mode (bool): Whether to use the latent space or the image space.
        n_train_timesteps (int): Number of training timesteps.
        guidance_scale (int): Scale of the guidance signal.
    """

    def __init__(
        self,
        model_name: str = "CompVis/stable-diffusion-v1-4",
        latent_mode: bool = True,
        n_train_timesteps: int = 1000,
        guidance_scale: int = 100,
    ) -> None:
        super(StableDiffusionSDS, self).__init__()

        # Parameters
        self.guidance_scale = guidance_scale
        self.latent_mode = latent_mode
        self.n_train_timesteps = n_train_timesteps
        self.min_step = int(self.n_train_timesteps * 0.02)
        self.max_step = int(self.n_train_timesteps * 0.98)

        # Components
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", use_auth_token=self.token
        )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", use_auth_token=self.token
        )
        self.image_encoder = None
        self.image_processor = None

        # Scheduler
        self.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=self.num_train_timesteps,
        )
        self.alphas = self.scheduler.alphas_cumprod

    @torch.inference_mode()
    def get_text_embeddings(self, prompt: str):
        """
        Get CLIP text embeddings for a given prompt.

        Args:
            prompt (str): Prompt to encode.

        Returns:
            torch.tensor: CLIP text embeddings.
        """
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_latents_gradient(
        self, text_embeddings: torch.tensor, latents: torch.tensor
    ) -> torch.tensor:
        """
        Get the gradient of the latent code of the rendered image

        Args:
            text_embeddings (torch.tensor): CLIP text embeddings.
            latents (torch.tensor): Latent code of the rendered image.

        Returns:
            torch.tensor: Gradient of the latent code of the rendered image.
        """
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        with torch.inference_mode():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # Split noise prediction into unconditional and conditional components
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute loss and gradients
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        return grad

    def produce_latents(
        self,
        text_embeddings: torch.tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latent: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """
        Generate an image latent code for a given text embedding.

        Args:
            text_embeddings (torch.tensor): CLIP text embeddings.
            height (int, optional): Height of the image. Defaults to 512.
            width (int, optional): Width of the image. Defaults to 512.
            num_inference_steps (int, optional): Number of diffusion steps. Defaults to 50.
            guidance_scale (float, optional): Scale of the guidance signal. Defaults to 7.5.
            latent (Optional[torch.tensor], optional): Latent code of the rendered image.

        Returns:
            torch.tensor: Latent code of the generated image.
        """
        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def encode_text(self, prompt: str) -> torch.tensor:
        """
        Encode text with the CLIP text encoder.

        Args:
            prompt (str): Prompt to encode.

        Returns:
            torch.tensor: Encoded text.
        """
        # Tokenize and encode prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Add unconditional text embedding
        uncond_input = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Concatenate embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def encode_image(self, images: torch.tensor) -> torch.tensor:
        """
        Encode images with the VAE image encoder.

        Args:
            images (torch.tensor): Images as B x 3 x H x W tensor.

        Returns:
            torch.tensor: Encoded images.
        """
        imgages = 2 * imgages - 1

        posterior = self.vae.encode(imgages).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def decode_image(self, latents: torch.tensor) -> torch.tensor:
        """
        Decode images with the VAE image decoder.

        Args:
            latents (torch.tensor): Latent codes of the images.

        Returns:
            torch.tensor: Decoded images.
        """
        latents = 1 / 0.18215 * latents

        with torch.inference_mode():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs
