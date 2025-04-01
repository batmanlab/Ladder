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
    """
        Computes and saves the top-K sentences that explain error slices by comparing
        average CLIP-aligned embeddings of correct vs. incorrect classifier predictions.

        Args:
            df_fold_corr_indx: List of indices with correct predictions.
            df_fold_incorr_indx: List of indices with incorrect predictions.
            clf_image_emb_path: Path to classifier image embeddings.
            language_emb_path: Path to sentence language embeddings.
            aligner_path: Path to aligner model for mapping image to text space.
            sent_path: Path to list of original sentences.
            save_path: Directory to save output explanation sentences.
            topKsent: Number of top sentences to retrieve.
            diff_save_file, corr_save_file, incorr_save_file: Filenames for saving results.
    """

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
    """
    Processes the Waterbirds dataset to analyze prediction errors using aligned embeddings
    and retrieves natural language explanations for error slices.

    Args:
        save_path (Path): Directory to save results and output sentences.
        clf_results_csv (str): Path to CSV containing model predictions and ground truth.
        clf_image_emb_path (str): Path to classifier-generated image embeddings.
        language_emb_path (str): Path to sentence embeddings (.npy).
        aligner_path (str): Path to the aligner model file.
        sent_path (str): Path to the pickle file with text sentences.
        topKsent (int): Number of top sentences to extract per slice.
        prediction_col (str): Name of column in the CSV that contains model predictions.
        out_file (str, optional): Path to log the worst-group accuracy.
    """

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
    """
    Processes the MetaShift dataset to analyze prediction error slices and explain them using aligned sentence embeddings.

    Args:
        save_path (Path): Output directory to save explanation results.
        clf_results_csv (str): CSV containing classifier predictions and true labels.
        clf_image_emb_path (str): Path to image embeddings (.npy).
        language_emb_path (str): Path to sentence embeddings (.npy).
        aligner_path (str): Path to projection aligner model.
        sent_path (str): Path to the pickle file with text sentences.
        topKsent (int): Number of top-ranked sentences to extract.
        prediction_col (str): Column name in the CSV containing model prediction values.
    """
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
    """
    Handles error slice discovery and sentence explanations for CelebA dataset (e.g., Blonde vs. Non-Blonde).

    Args:
        save_path (Path): Output directory for saving sentence results.
        clf_results_csv (str): Path to CSV with classifier predictions and labels.
        clf_image_emb_path (str): Path to .npy image embeddings from classifier.
        language_emb_path (str): Path to .npy sentence embeddings.
        aligner_path (str): Path to the trained aligner model.
        sent_path (str): Path to .pkl file with original sentences.
        topKsent (int): Number of top sentences to extract.
        prediction_col (str): Column name containing predicted probabilities.
        out_file (str): Path to output log file.
    """

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


def discover_error_slices_via_sent_mammo(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col, out_file=None, dataset=None):
    """
        Explains classification performance on mammogram datasets (e.g., RSNA, VinDr) using aligned sentence embeddings.

        Args:
            save_path (Path): Where to save the output.
            clf_results_csv (str): Classifier prediction results CSV.
            clf_image_emb_path (str): Path to classifier embeddings.
            language_emb_path (str): Sentence embeddings path.
            aligner_path (str): Trained aligner model.
            sent_path (str): Sentence list pickle path.
            topKsent (int): Number of top-k sentences.
            prediction_col (str): Prediction column name.
            out_file (str): Log file for results.
            dataset (str): Dataset name (rsna or vindr).
    """
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
    if prediction_col == "out_put_predict":
        df['Predictions_bin'] = (df[prediction_col] >= 0.5).astype(int)
        pred_col = "Predictions_bin"
    else:
        pred_col = f"{prediction_col}_bin"

    cancer_df_corr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 1)].index.tolist()
    cancer_df_incorr_indx = df[(df["out_put_GT"] == 1) & (df[pred_col] == 0)].index.tolist()
    diff_save_file = f"cancer_error_top_{topKsent}_sent_diff_emb.txt" if dataset == "rsna" else f"abnormal_error_top_{topKsent}_sent_diff_emb.txt"
    corr_save_file = f"cancer_error_top_{topKsent}_sent_corr_emb.txt" if dataset == "rsna" else f"cancer_error_top_{topKsent}_sent_corr_emb.txt"
    incorr_save_file = f"cancer_error_top_{topKsent}_sent_incorr_emb.txt" if dataset == "rsna" else f"cancer_error_top_{topKsent}_sent_incorr_emb.txt"

    get_sentences_for_err_slices(
        cancer_df_corr_indx, cancer_df_incorr_indx, clf_image_emb_path, language_emb_path, aligner_path,
        sent_path, save_path, topKsent, diff_save_file=diff_save_file, corr_save_file=corr_save_file,
        incorr_save_file=incorr_save_file
    )


def discover_error_slices_via_sent_nih(
        save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
        prediction_col, out_file=None):
    """
        Discovers and explains Pneumothorax classification errors in the NIH dataset using aligned language embeddings.

        Args:
            save_path (Path): Directory to store results.
            clf_results_csv (str): Path to classifier results CSV.
            clf_image_emb_path (str): Path to image embeddings.
            language_emb_path (str): Path to language (sentence) embeddings.
            aligner_path (str): Path to projection aligner model file.
            sent_path (str): Path to .pkl sentence list.
            topKsent (int): Number of top-ranked explanatory sentences to extract.
            prediction_col (str): Name of prediction column in CSV.
            out_file (str, optional): File path to log accuracy info.
    """
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
    """
        Dispatcher function to call the appropriate dataset-specific sentence discovery logic.

        Args:
            dataset (str): Dataset name.
            save_path (Path): Directory to save outputs.
            clf_results_csv (str): Path to classifier results CSV.
            clf_image_emb_path (str): Path to classifier image embeddings.
            language_emb_path (str): Path to sentence embeddings.
            aligner_path (str): Path to projection aligner.
            sent_path (str): Path to .pkl file with sentences.
            topKsent (int): Number of top sentences to extract.
            prediction_col (str): Column in CSV for predictions.
            out_file (str, optional): Log file for output.
    """
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
        discover_error_slices_via_sent_mammo(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col, out_file=out_file, dataset=dataset.lower())

    elif dataset.lower() == "nih":
        discover_error_slices_via_sent_nih(
            save_path, clf_results_csv, clf_image_emb_path, language_emb_path, aligner_path, sent_path, topKsent,
            prediction_col=prediction_col, out_file=out_file)


def config():
    parser = argparse.ArgumentParser(description="Discover and explain classifier error slices using text embeddings.")
    parser.add_argument(
        "--dataset", default="Waterbirds", type=str, help="Name of the dataset to evaluate.")
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help="Path to save outputs including explanation sentences."
    )
    parser.add_argument(
        "--clf_results_csv", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv",
        help="CSV file containing predictions and labels."
    )
    parser.add_argument(
        "--clf_image_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy",
        help="Classifier-generated image embeddings (.npy)."
    )
    parser.add_argument(
        "--language_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/sent_emb_word.npy",
        help="Language embedding file (.npy) for sentences."
    )
    parser.add_argument(
        "--sent_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/sentences.pkl",
        help="Pickle file containing a list of text sentences."
    )
    parser.add_argument(
        "--aligner_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth",
        help="Path to pretrained projection aligner (image → text embedding space)."
    )
    parser.add_argument(
        "--topKsent", default="20", type=int, help="Top-K most similar sentences to extract.")
    parser.add_argument(
        "--prediction_col", default="out_put_predict", type=str,
        help="Column name for model prediction values.")
    parser.add_argument("--seed", default="0", type=int, help="Seed value for reproducibility.")

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
