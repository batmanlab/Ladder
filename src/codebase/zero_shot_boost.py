import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import clip

from model_factory import create_clip
from prompts.zs_prompts import get_vision_prompts, get_nih_prompts, get_rsna_prompts
from utils import seed_all
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import argparse
import os


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Waterbirds", type=str)
    parser.add_argument(
        "--data_csv",
        default="/ocean/projects/asc170022p/shg121/PhD/Multimodal-mistakes-debug/data/waterbirds/waterbird_complete95_forest2water2/rsna_metadata.csv",
        type=str
    )

    parser.add_argument(
        "--clip_reps_path",
        default="/ocean/projects/asc170022p/shg121/PhD/Multimodal-mistakes-debug/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_RN50/{1}_clip_embeddings.npy",
        type=str,
        help="Save path of clip (VLM) representations"
    )
    parser.add_argument(
        "--clip_check_pt",
        default="/ocean/projects/asc170022p/shg121/PhD/Multimodal-mistakes-debug/out/NIH_Cxrclip/resnet50/seed{}/swint_mc.tar",
        type=str,
        help="Checkpoint path of cxr-clip"
    )

    parser.add_argument("--fold", default=0, type=int, help="which fold?")
    parser.add_argument("--seed", default=0, type=int, help="which seed?")
    parser.add_argument("--batch_size", default=32, type=int, help="what batch size?")
    parser.add_argument("--prompt_type", default="baseline", type=str, help="baseline or slice")
    parser.add_argument("--arch", default="ResNet50", type=str, help="ResNet50 or ViT")
    return parser.parse_args()


def calculate_zs_accuracy_vision(loader, prompt, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    class_0_desc, class_1_desc = prompt[0], prompt[1]
    class_0_tokens = clip.tokenize(class_0_desc).to(device)
    class_1_tokens = clip.tokenize(class_1_desc).to(device)

    with torch.no_grad():
        class_0_features = model.encode_text(class_0_tokens).mean(dim=0, keepdim=True)
        class_1_features = model.encode_text(class_1_tokens).mean(dim=0, keepdim=True)

    total = 0
    correct = 0
    for img_reps, gt in tqdm(loader, desc="Computing Accuracy", leave=False):
        with torch.no_grad():
            img_reps = img_reps.to(device)
            gt = gt.to(device)
            logits = torch.cat([class_0_features, class_1_features]) @ img_reps.T
            probs = logits.softmax(dim=0)
            predictions = torch.argmax(probs, dim=0)

        correct += (predictions == gt).sum().item()
        total += gt.size(0)

    accuracy = correct / total
    return accuracy


def get_lang_emb_nih(disease_desc, clip_model):
    print(disease_desc)
    idx = 1
    language_emb = torch.FloatTensor()
    with tqdm(total=len(disease_desc)) as t:
        for sentence in disease_desc:
            prompts = [sentence]
            sentences_w_tube_token = clip_model["tokenizer"](
                prompts, padding="longest", truncation=True, return_tensors="pt",
                max_length=256
            )
            with torch.no_grad():
                text_emb = clip_model["model"].encode_text(sentences_w_tube_token.to(args.device))
                text_emb = clip_model["model"].text_projection(text_emb) if clip_model["model"].projection else text_emb
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)

            language_emb = torch.cat((language_emb, text_emb.detach().cpu()), dim=0)
            # print(language_emb.size())
            t.set_postfix(idx='{0}'.format(idx))
            t.update()
            idx += 1

    print(language_emb.size())
    disease_emb = language_emb.mean(dim=0)
    return disease_emb


def calculate_zs_accuracy_nih(loader, prompt, device):
    args.clip_vision_encoder = "swin-tiny-cxr-clip"
    args.device = device
    clip_model = create_clip(args)

    neg_disease_desc, pos_disease_desc = prompt[0], prompt[1]
    neg_disease_emb = get_lang_emb_nih(neg_disease_desc, clip_model).unsqueeze(dim=0).to(device)
    pos_disease_emb = get_lang_emb_nih(pos_disease_desc, clip_model).unsqueeze(dim=0).to(device)
    print(neg_disease_emb.shape, pos_disease_emb.shape)

    total = 0
    correct = 0
    for img_reps, gt in tqdm(loader, desc="Computing Accuracy", leave=False):
        with torch.no_grad():
            img_reps = img_reps.to(device)
            gt = gt.to(device)
            logits = img_reps @ torch.cat([neg_disease_emb, pos_disease_emb], dim=0).T
            probs = logits.softmax(dim=1)
            predictions = torch.argmax(probs, dim=1)

        correct += (predictions == gt).sum().item()
        total += gt.size(0)

    accuracy = correct / total
    return accuracy


def main(args):
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_reps = torch.from_numpy(np.load(args.clip_reps_path.format(args.seed, "test")))
    df = pd.read_csv(args.data_csv)
    if args.dataset.lower() == "waterbirds" or args.dataset.lower() == "celeba" or args.dataset.lower() == "metashift":
        gt = torch.from_numpy(df[df["split"] == 2]["y"].values)
        prompt = get_vision_prompts(args.dataset, args.prompt_type, args.arch, args.fold)
        print(prompt)
        dataset = TensorDataset(img_reps, gt)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Loaded dataset with {len(dataset)} samples")
        accuracy = calculate_zs_accuracy_vision(loader, prompt, device)
        print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")
        print("\n")

    elif args.dataset.lower() == "nih":
        gt = torch.from_numpy(df[df["val_train_split"] == 0]["Pneumothorax"].values)
        prompt = get_nih_prompts(args.prompt_type, args.arch, args.fold)
        dataset = TensorDataset(img_reps, gt)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Loaded dataset with {len(dataset)} samples")
        accuracy = calculate_zs_accuracy_nih(loader, prompt, device)
        print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")
        print("\n")

    elif args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr" or args.dataset.lower() == "embed":
        print(df.columns)
        gt = torch.from_numpy(df["out_put_GT"].values)
        prompt = get_rsna_prompts(args.prompt_type, args.arch, args.fold, args.dataset.lower())
        dataset = TensorDataset(img_reps, gt)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Loaded dataset with {len(dataset)} samples")
        accuracy = calculate_zs_accuracy_nih(loader, prompt, device)
        print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")
        print("\n")



if __name__ == "__main__":
    args = config()
    main(args)
