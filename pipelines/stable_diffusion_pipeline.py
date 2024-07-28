from diffusers import StableDiffusionPipeline

def create_stable_diffusion_pipeline(text_encoder, vae, unet, scheduler):
    return StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler
    )