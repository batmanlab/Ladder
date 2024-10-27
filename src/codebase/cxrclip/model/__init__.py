from typing import Dict

from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from .clip import CXRClip


def build_model(model_config: Dict, loss_config: Dict, tokenizer: PreTrainedTokenizer = None) -> nn.Module:
    if model_config["name"].lower() == "clip_custom":
        model = CXRClip(model_config, loss_config, tokenizer)
    else:
        raise KeyError(f"Not supported model: {model_config['name']}")
    return model
