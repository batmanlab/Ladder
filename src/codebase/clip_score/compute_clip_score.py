import numpy as np
import pandas as pd
import torch

from breastclip.data.data_utils import load_tokenizer
from breastclip.model import BreastClip
from cxrclip.model import CXRClip, build_model


def get_sentences_for_err_slices(
        df_fold_corr_indx, df_fold_incorr_indx, clf_image_emb_path, aligner_path,
        keyword, clip_model, tokenizer, device
):
    prompts = [keyword]
    sentences_w_tube_token = tokenizer(
        prompts, padding="longest", truncation=True, return_tensors="pt",
        max_length=256
    )
    with torch.no_grad():
        text_emb = clip_model.encode_text(sentences_w_tube_token.to(device))
        text_emb = clip_model.text_projection(text_emb) if clip_model.projection else text_emb
        text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)

    keyword_features = text_emb
    img_emb_clf = np.load(clf_image_emb_path)
    img_emb_clf_corr = img_emb_clf[df_fold_corr_indx]
    img_emb_clf_incorr = img_emb_clf[df_fold_incorr_indx]

    aligner = torch.load(aligner_path)
    W = aligner["W"].to(device)
    b = aligner["b"].to(device)

    img_emb_clf_corr_tensor = torch.from_numpy(img_emb_clf_corr).to(device)
    img_emb_clip_corr_tensor = img_emb_clf_corr_tensor @ W.T + b

    img_emb_clip_corr_tensor = img_emb_clip_corr_tensor.float()
    keyword_features = keyword_features.float()
    sim_corr = torch.matmul(img_emb_clip_corr_tensor, keyword_features.T)
    sim_corr_mean = torch.mean(sim_corr, dim=0)

    img_emb_clf_incorr_tensor = torch.from_numpy(img_emb_clf_incorr).to(device)
    img_emb_clip_incorr_tensor = img_emb_clf_incorr_tensor @ W.T + b
    img_emb_clip_incorr_tensor = img_emb_clip_incorr_tensor.float()
    keyword_features = keyword_features.float()
    sim_incorr = torch.matmul(img_emb_clip_incorr_tensor, keyword_features.T)
    sim_incorr_mean = torch.mean(sim_incorr, dim=0)
    clip_score = sim_corr_mean - sim_incorr_mean
    return clip_score.item() * 100


def compute_nih_clip_score():
    clip_check_pt = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/NIH_Cxrclip/resnet50/seed0/swint_mc.tar"
    ckpt = torch.load(clip_check_pt, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ckpt["config"]
    cfg["tokenizer"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/tokenizers"
    cfg["model"]["text_encoder"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/"
    print(cfg)
    model_config = cfg["model"]
    loss_config = cfg["loss"]
    tokenizer_config = cfg["tokenizer"]
    tokenizer_config[
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/tokenizers"
    tokenizer = load_tokenizer(**tokenizer_config)
    model_config["text_encoder"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface"
    clip_model = CXRClip(model_config, loss_config, tokenizer)
    clip_model = clip_model.to(device)
    ret = clip_model.load_state_dict(ckpt["model"], strict=False)
    print(ret)
    print("CLIP is loaded successfully")
    clip_model.eval()

    clf_image_emb_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/test_classifier_embeddings.npy"
    aligner_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/aligner_200.pth"
    clf_results_csv = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/test_additional_info.csv"
    result = {}
    # change the keywords here based on the extracted hypotheses
    data = ['loculated pneumothorax', 'chest tubes', 'fluid levels', 'size and extent of pneumothorax',
            'side of the body']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_col = "Predictions_bin"
    df = pd.read_csv(clf_results_csv)
    df[pred_col] = (df["out_put_predict"] >= 0.5).astype(int)
    df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()

    for keyword in data:
        clip_score = get_sentences_for_err_slices(
            df_corr_indx, df_incorr_indx, clf_image_emb_path, aligner_path,
            keyword.lower(), clip_model, tokenizer, device
        )
        result[keyword] = clip_score

    print(result)


def compute_rsna_clip_score():
    clip_check_pt = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(clip_check_pt, map_location="cpu")
    cfg = ckpt["config"]
    cfg["tokenizer"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/tokenizers"
    cfg["model"]["text_encoder"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/"
    tokenizer_config = cfg["tokenizer"]
    tokenizer = load_tokenizer(**tokenizer_config) if tokenizer_config is not None else None
    clip_model = BreastClip(cfg["model"], cfg["loss"], tokenizer)
    clip_model = clip_model.to(device)
    ret = clip_model.load_state_dict(ckpt["model"], strict=False)
    print(ret)
    print("CLIP is loaded successfully")
    clip_model.eval()

    clf_image_emb_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/RSNA/Neurips/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy"
    aligner_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/RSNA/Neurips/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth"
    clf_results_csv = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/RSNA/Neurips/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv"
    result = {}
    # change the keywords here based on the extracted hypotheses
    data = ['Scattered calcifications', 'Bilateral occurrences',
            'Multiple densities', 'Vascular calcifications', 'Benign appearances']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_col = "Predictions_bin"
    df = pd.read_csv(clf_results_csv)
    df[pred_col] = (df["out_put_predict"] >= 0.5).astype(int)
    df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()

    for keyword in data:
        clip_score = get_sentences_for_err_slices(
            df_corr_indx, df_incorr_indx, clf_image_emb_path, aligner_path,
            keyword.lower(), clip_model, tokenizer, device
        )
        result[keyword] = clip_score

    print(result)


def compute_vindr_clip_score():
    clip_check_pt = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(clip_check_pt, map_location="cpu")
    cfg = ckpt["config"]
    cfg["tokenizer"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/tokenizers"
    cfg["model"]["text_encoder"][
        "cache_dir"] = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/tokenizers/scc/huggingface/"
    tokenizer_config = cfg["tokenizer"]
    tokenizer = load_tokenizer(**tokenizer_config) if tokenizer_config is not None else None
    clip_model = BreastClip(cfg["model"], cfg["loss"], tokenizer)
    clip_model = clip_model.to(device)
    ret = clip_model.load_state_dict(ckpt["model"], strict=False)
    print(ret)
    print("CLIP is loaded successfully")
    clip_model.eval()

    clf_image_emb_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/ViNDr/Neurips/fold0/cancer/e2/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy"
    aligner_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/ViNDr/Neurips/fold0/cancer/e2/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth"
    clf_results_csv = "/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/ViNDr/Neurips/fold0/cancer/e2/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv"
    result = {}

    # change the keywords here based on the extracted hypotheses
    data = ['postsurgical changes', 'nodules', 'scattered calcifications', 'progressive calcifications', ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_col = "Predictions_bin"
    df = pd.read_csv(clf_results_csv)
    df[pred_col] = (df["out_put_predict"] >= 0.5).astype(int)
    df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()

    for keyword in data:
        clip_score = get_sentences_for_err_slices(
            df_corr_indx, df_incorr_indx, clf_image_emb_path, aligner_path,
            keyword.lower(), clip_model, tokenizer, device
        )
        result[keyword] = clip_score

    print(result)


if __name__ == "__main__":
    dataset = "vindr"
    if dataset == "nih":
        compute_nih_clip_score()
    elif dataset == "rsna":
        compute_rsna_clip_score()
    elif dataset == "vindr":
        compute_vindr_clip_score()
