from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation
import torch

def get_hf_model(name):
    
    # Load model and feature extractor
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(name)

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model.to(device)
    
    return seg_model, feature_extractor