from transformers import SegformerForSemanticSegmentation
import torch
import os
import yaml
import sys

def get_model(n_classes=None, pretrained_name="nvidia/segformer-b0-finetuned-ade-512-512",device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if n_classes is None:
        configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
        CONFIG_FILE_PATH = os.path.join(configs_dir, "Segformer.yaml")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = yaml.safe_load(f)
        n_classes = config['num_classes']

    model= SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name,
        num_labels=n_classes,
        ignore_mismatched_sizes=True)
    model = model.to(device)
    return model
    
if __name__ == "__main__":
    model=get_model()
    print("Model loaded successfully!")