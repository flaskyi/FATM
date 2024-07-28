import torch
import logging
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DiffusionPipelineTrainer, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from datasets import load_dataset
from accelerate import Accelerator
from huggingface_hub import push_to_hub

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    logger.info("Using GPU")
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
else:
    logger.warning("Using CPU, training will be slow")

text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

dataset = load_dataset("flaskyi/flaskyi-v1-dataset")

pipe = StableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    scheduler=scheduler
)

trainer = DiffusionPipelineTrainer(
    pipe,
    dataset=dataset,
    learning_rate=1e-5,
    train_batch_size=4,
    output_dir="./results"
)

logger.info("Starting training")
trainer.train()
logger.info("Training finished")

pipe.save_pretrained("flaskyi/flaskyi-v1")
logger.info("Model saved")

push_to_hub(
    repo_id="flaskyi/flaskyi-v1",
    local_dir="flaskyi/flaskyi-v1"
)
logger.info("Model pushed to Hugging Face")