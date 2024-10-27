import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metrics_factory.calculate_worst_group_acc import calculate_worst_group_acc_waterbirds, \
    calculate_worst_group_acc_rsna_mammo, calculate_worst_group_acc_celebA, \
    calculate_worst_group_acc_metashift, calculate_worst_group_acc_chexpert_no_findings, \
    calculate_worst_group_acc_med_img
from utils import seed_all

warnings.filterwarnings("ignore")
import argparse
import os


def get_sentences_for_err_slices(
        df_fold_corr_indx, df_fold_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path, sent_path,
        save_path, topKsent, diff_save_file, corr_save_file, incorr_save_file):
    img_emb_clf = np.load(clf_image_emb_path)
    img_emb_clf_corr = img_emb_clf[df_fold_corr_indx]
    img_emb_clf_incorr = img_emb_clf[df_fold_incorr_indx]

    print(f"\nCorrect: {len(df_fold_corr_indx)}, InCorrect: {len(df_fold_incorr_indx)}")
    print("Clf Shapes:")
    print(f"img_emb_clf: {img_emb_clf.shape}")
    print(f"img_emb_clf_corr: {img_emb_clf_corr.shape}, img_emb_clf_incorr: {img_emb_clf_incorr.shape}")

    print("==============> Aligner weights and biases <================")
    print(aligner_path)

    aligner = torch.load(aligner_path)
    W = aligner["W"]
    b = aligner["b"]

    print(f"\nW: {W[0][0:10]}, b: {b[0:10]}")

    img_emb_clf_corr_tensor = torch.from_numpy(img_emb_clf_corr)
    img_emb_clip_corr_tensor = img_emb_clf_corr_tensor @ W.T + b
    img_emb_clip_corr_np = img_emb_clip_corr_tensor.numpy()

    img_emb_clf_incorr_tensor = torch.from_numpy(img_emb_clf_incorr)
    img_emb_clip_incorr_tensor = img_emb_clf_incorr_tensor @ W.T + b
    img_emb_clip_incorr_np = img_emb_clip_incorr_tensor.numpy()

    print(f"Shapes after projecting to clip:")
    print(f"\nCorrect: {img_emb_clip_corr_np.shape}, InCorrect: {img_emb_clip_incorr_np.shape}")

    mean_img_emb_correct = torch.mean(img_emb_clip_corr_tensor, dim=0)
    mean_img_emb_incorrect = torch.mean(img_emb_clip_incorr_tensor, dim=0)
    print(f"\nMean Correct: {mean_img_emb_correct.size()}, Mean InCorrect: {mean_img_emb_incorrect.size()}")

    print("######################################################################")
    print("Top sentences using the difference of correct and incorrect embeddings:")

    diff_embedding = mean_img_emb_correct - mean_img_emb_incorrect
    print(diff_embedding.size())
    print(language_emb_path)

    report_emb = torch.tensor(np.load(language_emb_path))
    print(report_emb.size())

    report_emb = report_emb.to(torch.float32)
    diff_embedding = diff_embedding.to(torch.float32)
    sent_sim = torch.matmul(report_emb, diff_embedding)
    print(sent_sim.size())

    sent_sim_idx = torch.topk(sent_sim, k=topKsent, dim=0).indices.squeeze()

    report_sent = pickle.load(open(sent_path, "rb"))
    print(f"Length of report sentences: {len(report_sent)}")

    print("\n ")
    sent = [report_sent[i] for i in sent_sim_idx]
    big_sentence = ""
    for i, sentence in enumerate(sent, start=1):
        big_sentence += f"{i}. {sentence}\n"
        print(f"{i}. {sentence}")

    with open(save_path / diff_save_file, "w") as file:
        file.write(big_sentence)


def discover_error_slices_via_sent_waterbirds(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col, out_file=None):
    df = pd.read_csv(clf_results_csv)
    if prediction_col == "out_put_predict":
        df['Predictions_bin'] = (df[prediction_col] >= 0.5).astype(int)
        pred_col = "Predictions_bin"
    else:
        pred_col = prediction_col
    print(f"Prediction column: {pred_col}")
    calculate_worst_group_acc_waterbirds(df, pred_col=pred_col, attribute_col="attribute_bg_predict", log_file=out_file)

    print("\n")
    print("#############" * 10)
    print(
        "####################################### Waterbird error slices #######################################")
    waterbird_df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    waterbird_df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()
    get_sentences_for_err_slices(
        waterbird_df_corr_indx, waterbird_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"waterbirds_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"waterbirds_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"waterbirds_error_top_{topKsent}_sent_incorr_emb.txt"
    )

    print("\n")
    print("#############" * 10)
    print(
        "####################################### Landbird error slices #######################################")
    landbird_df_corr_indx = df[(df["out_put_GT"] == 0) & (df[pred_col] == 0)].index.tolist()
    landbird_df_incorr_indx = df[(df["out_put_GT"] == 0) & (df[pred_col] == 1)].index.tolist()
    get_sentences_for_err_slices(
        landbird_df_corr_indx, landbird_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"landbirds_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"landbirds_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"landbirds_error_top_{topKsent}_sent_incorr_emb.txt"
    )


def discover_error_slices_via_sent_metashift(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col):
    df = pd.read_csv(clf_results_csv)
    if prediction_col == "out_put_predict":
        df['Predictions_bin'] = (df[prediction_col] >= 0.5).astype(int)
        pred_col = "Predictions_bin"
    else:
        pred_col = prediction_col
    print(f"Prediction column: {pred_col}")
    calculate_worst_group_acc_metashift(df, pred_col=pred_col, attribute_col="attribute_bg_predict")

    print("\n")
    print("#############" * 10)
    print("####################################### Cat error slices #######################################")
    cat_df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    cat_df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()
    get_sentences_for_err_slices(
        cat_df_corr_indx, cat_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"cat_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"cat_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"cat_error_top_{topKsent}_sent_incorr_emb.txt"
    )

    print("\n")
    print("#############" * 10)
    print("####################################### Dog error slices #######################################")
    dog_df_corr_indx = df[(df["out_put_GT"] == 0) & (df[pred_col] == 0)].index.tolist()
    dog_df_incorr_indx = df[(df["out_put_GT"] == 0) & (df[pred_col] == 1)].index.tolist()
    get_sentences_for_err_slices(
        dog_df_corr_indx, dog_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"dog_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"dog_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"dog_error_top_{topKsent}_sent_incorr_emb.txt"
    )


def discover_error_slices_via_sent_celeba(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col, out_file):
    df = pd.read_csv(clf_results_csv)
    if prediction_col == "out_put_predict":
        df['Predictions_bin'] = (df[prediction_col] >= 0.5).astype(int)
        pos_pred_col = "Predictions_bin"
        neg_pred_col = "Predictions_bin"
    else:
        pos_pred_col = "prediction_col"
        neg_pred_col = "prediction_col"
    print(f"Prediction column: {pos_pred_col, neg_pred_col}")
    calculate_worst_group_acc_celebA(
        df, pos_pred_col, neg_pred_col, attribute_col="attribute_bg_predict", log_file=out_file)

    print("\n")
    print("#############" * 10)
    print("####################################### Blonde error slices #######################################")
    celebA_df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pos_pred_col] == 1)].index.tolist()
    celebA_df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[neg_pred_col] == 0)].index.tolist()
    get_sentences_for_err_slices(
        celebA_df_corr_indx, celebA_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"celebA_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"celebA_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"celebA_error_top_{topKsent}_sent_incorr_emb.txt"
    )


def discover_error_slices_via_sent_rsna(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col, out_file=None):
    df = pd.read_csv(clf_results_csv)
    pos_pred_col = "out_put_predict"
    neg_pred_col = "out_put_predict"

    print(f"Prediction columns: {pos_pred_col, neg_pred_col}")
    acc_cancer_wo_calc = calculate_worst_group_acc_med_img(
        df, pos_pred_col=pos_pred_col, neg_pred_col=neg_pred_col, attribute_col="calc", log_file=out_file,
        disease="Cancer")
    print(f"Avg. accuracy worst group: {acc_cancer_wo_calc}")

    print("\n")
    print("#############" * 10)
    print("####################################### Cancer error slices #######################################")
    cancer_df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pos_pred_col] == 1)].index.tolist()
    cancer_df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[neg_pred_col] == 0)].index.tolist()
    print(xxxxxx)
    get_sentences_for_err_slices(
        cancer_df_corr_indx, cancer_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"cancer_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"cancer_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"cancer_error_top_{topKsent}_sent_incorr_emb.txt"
    )


def discover_error_slices_via_sent_nih(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col, out_file=None):
    df = pd.read_csv(clf_results_csv)
    pos_pred_col = "out_put_predict"
    neg_pred_col = "out_put_predict"
    print(f"Prediction columns: {pos_pred_col, neg_pred_col}")
    acc_pneumothorax_wo_tube = calculate_worst_group_acc_med_img(
        df, pos_pred_col=pos_pred_col, neg_pred_col=neg_pred_col, attribute_col="tube", log_file=out_file,
        disease="Pneumothorax")
    print(f"Avg. accuracy worst group: {acc_pneumothorax_wo_tube}")
    print("\n")
    print("#############" * 10)
    print("##################################### NIH Pneumothorax error slices #####################################")
    if prediction_col == "out_put_predict":
        df['Predictions_bin'] = (df[prediction_col] >= 0.5).astype(int)
        pred_col = "Predictions_bin"
    else:
        pred_col = f"{prediction_col}_bin"
    df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()
    get_sentences_for_err_slices(
        df_corr_indx, df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent,
        diff_save_file=f"pneumothorax_error_top_{topKsent}_sent_diff_emb.txt",
        corr_save_file=f"pneumothorax_error_top_{topKsent}_sent_corr_emb.txt",
        incorr_save_file=f"pneumothorax_error_top_{topKsent}_sent_incorr_emb.txt"
    )



def discover_error_slices_via_sent(
        dataset, save_path, clf_results_csv, clf_image_emb_path, language_emb_path,
        aligner_path, sent_path, topKsent, prediction_col="out_put_predict", out_file=None):
    if dataset.lower() == "waterbirds":
        discover_error_slices_via_sent_waterbirds(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col, out_file=out_file)

    elif dataset.lower() == "celeba":
        discover_error_slices_via_sent_celeba(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col, out_file=out_file)

    elif dataset.lower() == "metashift":
        discover_error_slices_via_sent_metashift(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col)

    elif dataset.lower() == "rsna" or dataset.lower() == "vindr":
        discover_error_slices_via_sent_rsna(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col, out_file=out_file)

    elif dataset.lower() == "nih":
        discover_error_slices_via_sent_nih(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col, out_file=out_file)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Waterbirds", type=str)
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help=""
    )
    parser.add_argument(
        "--clf_results_csv", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv",
        help=""
    )
    parser.add_argument(
        "--clf_image_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy",
        help=""
    )
    parser.add_argument(
        "--language_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/sent_emb_word.npy",
        help=""
    )
    parser.add_argument(
        "--sent_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/sentences.pkl",
        help=""
    )
    parser.add_argument(
        "--aligner_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth",
        help=""
    )
    parser.add_argument("--topKsent", default="20", type=int)
    parser.add_argument("--prediction_col", default="out_put_predict", type=str)
    parser.add_argument("--seed", default="0", type=int)

    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.save_path = Path(args.save_path.format(args.seed))
    args.clf_results_csv = args.clf_results_csv.format(args.seed)
    args.clf_image_emb_path = args.clf_image_emb_path.format(args.seed)
    args.language_emb_path = args.language_emb_path.format(args.seed)
    args.aligner_path = args.aligner_path.format(args.seed)
    args.sent_path = args.sent_path.format(args.seed)

    args.save_path.mkdir(parents=True, exist_ok=True)
    out_file = args.save_path / "ladder_discover_slices_performance_ERM.txt"
    print("\n")
    print(args.save_path)
    discover_error_slices_via_sent(
        args.dataset, args.save_path, args.clf_results_csv, args.clf_image_emb_path, args.language_emb_path,
        args.aligner_path, args.sent_path, args.topKsent, args.prediction_col, out_file
    )
    print("Completed")
    print(f"Performance information at: {out_file}")


if __name__ == "__main__":
    _args = config()
    main(_args)
