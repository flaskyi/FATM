from diffusers import DiffusionPipelineTrainer

def train_pipeline(pipe, dataset):
    trainer = DiffusionPipelineTrainer(
        pipe,
        dataset=dataset,
        learning_rate=1e-5,
        train_batch_size=4,
        output_dir="./results"
    )
    trainer.train()