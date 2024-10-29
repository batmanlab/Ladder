from typing import Dict

from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from .clip import BreastClip
from .image_classification import MammoClassification
from .mamo_efficient_net import MammoEfficientNet
from .modules import EfficientNet


def build_model(model_config: Dict, loss_config: Dict, tokenizer: PreTrainedTokenizer = None) -> nn.Module:
    if model_config["name"].lower() == "clip_custom":
        model = BreastClip(model_config, loss_config, tokenizer)
    elif model_config["name"].lower() == "finetune_classification":
        model_type = model_config["image_encoder"]["model_type"] if "model_type" in model_config[
            "image_encoder"] else "vit"
        model = MammoClassification(model_config, model_type)
    elif model_config["name"].lower() == "pretrained_classifier" and model_config[
        "clf_arch"].lower() == "tf_efficientnet_b5_ns-detect":
        model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
    elif model_config["name"].lower() == "pretrained_classifier" and model_config[
        "clf_arch"].lower() == "tf_efficientnetv2-detect":
        model = EfficientNet.from_pretrained("efficientnet-b2", num_classes=1)
    else:
        raise KeyError(f"Not supported model: {model_config['name']}")
    return model
