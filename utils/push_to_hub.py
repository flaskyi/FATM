from huggingface_hub import push_to_hub

def push_model_to_hub(repo_id, local_dir):
    push_to_hub(
        repo_id=repo_id,
        local_dir=local_dir
    )