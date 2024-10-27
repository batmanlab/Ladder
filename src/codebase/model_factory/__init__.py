import clip
import torch

# from breastclip.data.data_utils import load_tokenizer
# from breastclip.model import BreastClip
from . import networks
# from model_factory.efficientnet_custom import EfficientNet
# from model_factory.mammo_classifier import MammoClassifier
from .resnet_nih import ResNet

from cxrclip.data.data_utils import load_tokenizer
from cxrclip.model import CXRClip
from utils import get_hparams


def get_vision_clf(data_type, input_shape, num_classes, hparams, checkpoint, device, mode="eval"):
    feature_maps = networks.Featurizer(data_type, input_shape, hparams)
    classifier = networks.Classifier(
        in_features=feature_maps.n_outputs,
        out_features=num_classes,
        is_nonlinear=hparams['nonlinear_classifier']
    )
    if mode == "eval" or mode == "last_layer_finetune":
        feature_maps_state_dict = {key.replace('featurizer.network.', 'network.'): value for key, value in
                                   checkpoint.items()
                                   if key.startswith('featurizer.network')}
        ret = feature_maps.load_state_dict(feature_maps_state_dict)
        print(ret)
        print("feature_maps is loaded successfully")
        classifier_state_dict = {'weight': checkpoint['classifier.weight'], 'bias': checkpoint['classifier.bias']}
        ret = classifier.load_state_dict(classifier_state_dict)
        print(ret)
        print("classifier is loaded successfully")
        print(f"Pretrained model is loaded successfully")
        feature_maps.to(device).eval()
        classifier.to(device).eval()
    return feature_maps, classifier


def create_classifier(args, mode="eval"):
    if args.dataset.lower() == "waterbirds" or args.dataset.lower() == "celeba" or args.dataset.lower() == "metashift":
        hparams = get_hparams(args.dataset, args.classifier)
        print(hparams)
        data_type = "images"
        num_classes = 2
        checkpoint = torch.load(args.classifier_check_pt.format(args.seed))['model_dict']
        feature_maps, classifier = get_vision_clf(data_type, args.input_shape, num_classes, hparams,
                                                  checkpoint, args.device, mode)
        clf = {
            "feature_maps": feature_maps,
            "classifier": classifier
        }
        return clf
    elif args.dataset.lower() == "nih" and args.classifier.lower() == "resnet50":
        clf = ResNet(
            pre_trained=True, n_class=1, model_choice=args.classifier,
            weights=args.classifier_check_pt.format(args.seed), return_reps=True, mode=mode
        )
        clf = clf.to(args.device)
        if mode == "eval":
            clf.eval()
        print("NIH Classifier loaded successfully")
        return clf
    elif (
            args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr" or args.dataset.lower() == "embed"
    ) and args.classifier.lower() == "efficientnet-b5":
        num_classes = 1
        args.classifier_check_pt = args.classifier_check_pt.format(args.seed)
        clf = MammoClassifier(args.classifier, args.classifier_check_pt, num_classes)
        clf = clf.to(args.device)
        if mode == "eval":
            clf.eval()
        return clf


def create_clip(args):
    if args.dataset.lower() == "waterbirds" or args.dataset.lower() == "celeba" or args.dataset.lower() == "metashift":
        model, preprocess = clip.load(args.clip_vision_encoder, args.device)
        print("CLIP is loaded successfully")
        return {
            "model": model,
            "type": "vision_clip"
        }
    elif (
            args.dataset.lower() == "nih"
    ) and (args.clip_vision_encoder.lower() == "swin-tiny-cxr-clip_mc" or
           args.clip_vision_encoder.lower() == "swin-tiny-cxr-clip_mcc"):
        args.clip_check_pt = args.clip_check_pt.format(args.seed)
        ckpt = torch.load(args.clip_check_pt, map_location="cpu")
        cfg = ckpt["config"]
        cfg["tokenizer"]["cache_dir"] = args.tokenizers
        cfg["model"]["text_encoder"]["cache_dir"] = args.cache_dir
        model_config = cfg["model"]
        loss_config = cfg["loss"]
        tokenizer_config = cfg["tokenizer"]
        tokenizer_config["cache_dir"] = args.tokenizers
        tokenizer = load_tokenizer(**tokenizer_config)
        model_config["text_encoder"]["cache_dir"] = args.cache_dir
        model = CXRClip(model_config, loss_config, tokenizer)
        model = model.to(args.device)
        ret = model.load_state_dict(ckpt["model"], strict=False)
        print(ret)
        print("CLIP is loaded successfully")
        model.eval()
        return {
            "model": model,
            "tokenizer": tokenizer,
            "type": "cxr_clip",
        }
    elif args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr":
        ckpt = torch.load(args.clip_check_pt, map_location="cpu")
        cfg = ckpt["config"]
        tokenizer_config = cfg["tokenizer"]
        tokenizer_config["cache_dir"] = tokenizer_config["cache_dir"].replace(
            "/ocean/projects/asc170022p/shg121/PhD", "/restricted/projectnb/batmanlab/shawn24/PhD")
        cfg["model"]["text_encoder"]["cache_dir"] = cfg["model"]["text_encoder"]["cache_dir"].replace(
            "/ocean/projects/asc170022p/shg121/PhD", "/restricted/projectnb/batmanlab/shawn24/PhD")
        print(tokenizer_config)

        tokenizer = load_tokenizer(**tokenizer_config) if tokenizer_config is not None else None
        model = BreastClip(cfg["model"], cfg["loss"], tokenizer)
        model = model.to(args.device)
        ret = model.load_state_dict(ckpt["model"], strict=False)
        print(ret)
        print("CLIP is loaded successfully")
        model.eval()
        return {
            "model": model,
            "type": "vision_clip",
            "tokenizer": tokenizer
        }
