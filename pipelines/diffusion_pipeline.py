import torch
from diffusers import DiffusionPipeline

def load_diffusion_pipelines():
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0_Turbo",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16"
    )
    pipe2 = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V2.02_Turbo",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16"
    )
    return pipe, pipe2