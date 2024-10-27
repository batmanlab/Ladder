import torch
from torch import nn
from transformers import AutoConfig, AutoModel, SwinModel, ViTModel
from breastclip.model.modules import load_image_encoder, LinearClassifier
import os


class HuggingfaceImageEncoder(nn.Module):
    def __init__(
            self,
            name: str = "google/vit-base-patch16-224",
            pretrained: bool = True,
            gradient_checkpointing: bool = False,
            cache_dir: str = "~/.cache/huggingface/hub",
            model_type: str = "vit",
            local_files_only: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        if pretrained:
            if self.model_type == "swin":
                self.image_encoder = SwinModel.from_pretrained(name)
            else:
                self.image_encoder = AutoModel.from_pretrained(
                    name, add_pooling_layer=False, cache_dir=cache_dir, local_files_only=local_files_only
                )
        else:
            # initializing with a config file does not load the weights associated with the model
            model_config = AutoConfig.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)
            if type(model_config).__name__ == "ViTConfig":
                self.image_encoder = ViTModel(model_config, add_pooling_layer=False)
            else:
                # TODO: add vision models if needed
                raise NotImplementedError(f"Not support training from scratch : {type(model_config).__name__}")

        if gradient_checkpointing and self.image_encoder.supports_gradient_checkpointing:
            self.image_encoder.gradient_checkpointing_enable()

        self.out_dim = self.image_encoder.config.hidden_size

    def forward(self, image):
        if self.model_type == "vit":
            output = self.image_encoder(pixel_values=image, interpolate_pos_encoding=True)
        elif self.model_type == "swin":
            output = self.image_encoder(pixel_values=image)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)


class SwinClassifier(nn.Module):
    def __init__(self, args, n_class):
        super(SwinClassifier, self).__init__()
        config_image_encoder = {
            "name": args.swin_encoder,
            "pretrained": args.pretrained_swin_encoder,
            "model_type": args.swin_model_type,
            "cache_dir": args.chk_pt_path / "swin_chk_pt"
        }

        cache_dir = config_image_encoder[
            "cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder[
                "gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = "swin"
        self.image_encoder = HuggingfaceImageEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )

        self.classifier = LinearClassifier(feature_dim=self.image_encoder.out_dim, num_class=n_class)

    def encode_image(self, image):
        image_features = self.image_encoder(image)
        global_features = image_features[:, 0]
        return global_features

    def forward(self, images):
        # get image features and predict
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits
