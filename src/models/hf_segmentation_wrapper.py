from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    SegformerConfig,
)
import torch


def get_hf_model(name, num_labels=2):

    # Load model and feature extractor
    image_processor = AutoImageProcessor.from_pretrained(name)
    config = SegformerConfig.from_pretrained(name, num_labels=num_labels)
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(
        name, config=config, ignore_mismatched_sizes=True
    )

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model.to(device)

    return seg_model, image_processor
