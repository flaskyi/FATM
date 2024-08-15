import torch
import logging
from datasets import load_dataset
from pipelines.diffusion_pipeline import load_diffusion_pipelines
from pipelines.stable_diffusion_pipeline import create_stable_diffusion_pipeline
from training.trainer import train_pipeline
from utils.push_to_hub import push_model_to_hub
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    logger.info("Using GPU")
    pipe, pipe2 = load_diffusion_pipelines()
else:
    logger.warning("Using CPU, training will be slow")

text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

dataset = load_dataset("flaskyi/flaskyi-v1-dataset")

pipe = create_stable_diffusion_pipeline(text_encoder, vae, unet, scheduler)

logger.info("Starting training")
train_pipeline(pipe, dataset)
logger.info("Training finished")

pipe.save_pretrained("flaskyi/flaskyi-v1")
logger.info("Model saved")

push_model_to_hub("flaskyi/flaskyi-v1", "./results")
logger.info("Model pushed to Hugging Face")