import numpy as np
import pandas as pd
import torchvision
from sklearn.metrics import roc_auc_score

from dataset_factory import UrbanCars
from metrics_factory import auroc, MultiDimAverageMeter
import logging
import os
import torch
from tqdm import tqdm


def calculate_worst_group_acc_waterbirds(
        df, pred_col, attribute_col="attribute_bg_predict", filter_waterbirds=True, filter_landbirds=True,
        log_file=None
):
    df[pred_col] = (df[pred_col] >= 0.5).astype(int)
    acc_waterbirds_land = 0
    acc_landbirds_water = 0
    if attribute_col == "attribute_bg_predict":
        concept_1 = "water"
        concept_2 = "land"
    else:
        concept_1 = f"w_{attribute_col}"
        concept_2 = f"w/o_{attribute_col}"

    if filter_waterbirds:
        print("#############" * 10)
        print("####################################### Waterbirds Stats #######################################")

        waterbirds_df = df[df["out_put_GT"] == 1]
        waterbirds_df_water_bg = waterbirds_df[waterbirds_df[attribute_col] == 1]
        waterbirds_df_land_bg = waterbirds_df[waterbirds_df[attribute_col] == 0]
        print(f"GT Shape Waterbirds on {concept_1} bg: {waterbirds_df_water_bg.shape}")
        print(
            f"Pred Shape Waterbirds on {concept_1} bg: {waterbirds_df_water_bg[waterbirds_df_water_bg[pred_col] == 1].shape}"
        )
        acc_waterbirds_water = waterbirds_df_water_bg[waterbirds_df_water_bg[pred_col] == 1].shape[0] / \
                               waterbirds_df_water_bg.shape[0]
        print(f"acc Waterbirds on {concept_1} bg: {acc_waterbirds_water}")

        print(f"GT Shape Waterbirds on {concept_2} bg: {waterbirds_df_land_bg.shape}")
        print(
            f"Pred Shape Waterbirds on {concept_2} bg: {waterbirds_df_land_bg[waterbirds_df_land_bg[pred_col] == 1].shape}"
        )
        acc_waterbirds_land = waterbirds_df_land_bg[waterbirds_df_land_bg[pred_col] == 1].shape[0] / \
                              waterbirds_df_land_bg.shape[0]

        print(f"acc Waterbirds on {concept_2} bg: {acc_waterbirds_land}")
        print("#############" * 10)

    if filter_landbirds:
        print("#############" * 10)
        print("####################################### Landbirds Stats #######################################")
        landbirds_df = df[df["out_put_GT"] == 0]
        landbirds_df_water_bg = landbirds_df[landbirds_df[attribute_col] == 1]
        landbirds_df_land_bg = landbirds_df[landbirds_df[attribute_col] == 0]
        print(f"GT Shape Landbirds on {concept_1} bg: {landbirds_df_water_bg.shape}")
        print(
            f"Pred Shape Landbirds on {concept_1} bg: {landbirds_df_water_bg[landbirds_df_water_bg[pred_col] == 0].shape}"
        )
        acc_landbirds_water = landbirds_df_water_bg[landbirds_df_water_bg[pred_col] == 0].shape[0] / \
                              landbirds_df_water_bg.shape[0]
        print(f"acc Landbirds on {concept_1} bg: {acc_landbirds_water}")

        print(f"GT Shape Landbirds on {concept_2} bg: {landbirds_df_land_bg.shape}")
        print(
            f"Pred Shape Landbirds on {concept_2} bg: {landbirds_df_land_bg[landbirds_df_land_bg[pred_col] == 0].shape}"
        )
        acc_landbirds_land = landbirds_df_land_bg[landbirds_df_land_bg[pred_col] == 0].shape[0] / \
                             landbirds_df_land_bg.shape[0]

        print(f"acc Landbirds on {concept_2} bg: {acc_landbirds_land}")
        print("#############" * 10)

    # df_vertical = pd.concat([waterbirds_df_land_bg, landbirds_df_water_bg], axis=0).reset_index(drop=True)
    # worst = (df_vertical["out_put_GT"].values == df_vertical[pred_col].values).sum() / df_vertical.shape[0]
    # print(f"\n Avg worst group acc: {worst}")

    avg = (df["out_put_GT"].values == df[pred_col].values).sum() / df.shape[0]
    avg_wga = min(acc_waterbirds_land, acc_landbirds_water)
    print(f"Avg acc: {avg}")

    if filter_waterbirds and filter_landbirds:
        if log_file:
            with open(log_file, 'w') as f:
                print(f"Mean acc: {avg}", file=f)
                print(f"Worst group acc: {avg_wga}", file=f)

        print(f"Worst group acc: {avg_wga}")
        return avg_wga
    elif filter_waterbirds:
        if log_file:
            with open(log_file, 'w') as f:
                print(f"Mean acc: {acc_waterbirds_land}", file=f)
                print(f"Worst group acc: {avg_wga}", file=f)
        print(f"Avg worst group acc: {acc_waterbirds_land}")
        return acc_waterbirds_land
    elif filter_landbirds:
        if log_file:
            with open(log_file, 'w') as f:
                print(f"Mean acc: {acc_waterbirds_land}", file=f)
                print(f"Worst group acc: {avg_wga}", file=f)
        print(f"Avg worst group acc: {acc_landbirds_water}")
        return acc_landbirds_water


def calculate_worst_group_acc_metashift(
        df, pred_col, attribute_col="attribute_bg_predict", filter_cats=True, filter_dogs=True):
    df[pred_col] = (df[pred_col] >= 0.5).astype(int)
    print(f"df: {df.shape}")
    acc_cat_outdoor = 0
    acc_dogs_indoor = 0
    if attribute_col == "attribute_bg_predict":
        concept_1 = "indoor"
        concept_2 = "outdoor"
    else:
        concept_1 = f"w_{attribute_col}"
        concept_2 = f"w/o_{attribute_col}"

    if filter_cats:
        print("#############" * 10)
        print("####################################### Cat Stats #######################################")
        cat_df = df[df["out_put_GT"] == 1]
        cat_indoor = cat_df[cat_df[attribute_col] == 1]
        cat_outdoor = cat_df[cat_df[attribute_col] == 0]
        print(f"GT Shape cat ({concept_1}) bg: {cat_indoor.shape}")
        print(
            f"Pred Shape cat ({concept_1})r bg: {cat_indoor[cat_indoor[pred_col] == 1].shape}"
        )
        acc_cat_indoor = cat_indoor[cat_indoor[pred_col] == 1].shape[0] / cat_indoor.shape[0]
        print(f"acc cat ({concept_1}) bg: {acc_cat_indoor}")

        print(f"GT Shape cat ({concept_2}) bg: {cat_outdoor.shape}")
        print(
            f"Pred Shape cat ({concept_2}) bg: {cat_outdoor[cat_outdoor[pred_col] == 1].shape}"
        )
        acc_cat_outdoor = cat_outdoor[cat_outdoor[pred_col] == 1].shape[0] / cat_outdoor.shape[0]

        print(f"acc cat ({concept_2}) bg: {acc_cat_outdoor}")
        print("#############" * 10)

    print("\n")
    if filter_dogs:
        print("#############" * 10)
        print("####################################### Dog Stats #######################################")
        dog_df = df[df["out_put_GT"] == 0]
        dogs_indoor = dog_df[dog_df[attribute_col] == 1]
        dogs_outdoor = dog_df[dog_df[attribute_col] == 0]
        print(f"GT Shape dog ({concept_1}) bg: {dogs_indoor.shape}")
        print(
            f"Pred Shape dog ({concept_1}) bg: {dogs_indoor[dogs_indoor[pred_col] == 0].shape}"
        )
        acc_dogs_indoor = dogs_indoor[dogs_indoor[pred_col] == 0].shape[0] / dogs_indoor.shape[0]
        print(f"acc dog ({concept_1}) bg: {acc_dogs_indoor}")

        print(f"GT Shape dog ({concept_2}) bg: {dogs_outdoor.shape}")
        print(
            f"Pred Shape dog ({concept_2}) bg: {dogs_outdoor[dogs_outdoor[pred_col] == 0].shape}"
        )
        acc_dogs_outdoor = dogs_outdoor[dogs_outdoor[pred_col] == 0].shape[0] / \
                           dogs_outdoor.shape[0]

        print(f"acc dog ({concept_2}) bg: {acc_dogs_outdoor}")
        print("#############" * 10)

    avg = (df["out_put_GT"].values == df[pred_col].values).sum() / df.shape[0]
    print(f"Avg acc: {avg}")
    avg_wga = (acc_dogs_indoor + acc_cat_outdoor) / 2
    if filter_cats and filter_dogs:
        print(f"Avg worst group acc: {avg_wga}")
        return avg_wga
    elif filter_cats:
        print(f"Avg worst group acc: {acc_cat_outdoor}")
        return acc_cat_outdoor
    elif filter_dogs:
        print(f"Avg worst group acc: {acc_dogs_indoor}")
        return acc_dogs_indoor


def calculate_worst_group_acc_celebA(df, pos_pred_col, neg_pred_col, attribute_col="attribute_bg_predict",
                                     print_non_blonde=True, log_file=None):
    if attribute_col == "attribute_bg_predict":
        concept_1 = "male"
        concept_2 = "female"
    else:
        concept_1 = f"w_{attribute_col}"
        concept_2 = f"w/o_{attribute_col}"
    print("#############" * 10)
    print("####################################### CelebA Stats #######################################")
    blonde_df = df[df["out_put_GT"] == 1]
    blonde_male_df = blonde_df[blonde_df[attribute_col] == 1]
    blonde_female_df = blonde_df[blonde_df[attribute_col] == 0]
    print(f"GT Shape blonde {concept_1}: {blonde_male_df.shape}")
    print(
        f"Pred Shape blonde {concept_1}: {blonde_male_df[blonde_male_df[pos_pred_col] == 1].shape}"
    )
    acc_blonde_male = blonde_male_df[blonde_male_df[pos_pred_col] == 1].shape[0] / \
                      blonde_male_df.shape[0]
    print(f"acc blonde {concept_1}: {acc_blonde_male}")

    print(f"GT Shape blonde {concept_2}: {blonde_female_df.shape}")
    print(
        f"Pred Shape blonde {concept_2}: {blonde_female_df[blonde_female_df[pos_pred_col] == 1].shape}"
    )
    acc_blonde_female = blonde_female_df[blonde_female_df[pos_pred_col] == 1].shape[0] / \
                        blonde_female_df.shape[0]

    print(f"acc blonde {concept_2}: {acc_blonde_female}")
    print("#############" * 10)

    avg = (df["out_put_GT"].values == df[pos_pred_col].values).sum() / df.shape[0]
    print(f"==============>>> Avg acc: {avg} <<<==============")

    if attribute_col == "attribute_bg_predict":
        print(f"==============>>> Worst acc, blonde-{concept_1}: {acc_blonde_male} <<<==============")
        acc = acc_blonde_male
    else:
        print(f"==============>>> Worst acc, blonde-{concept_2}: {acc_blonde_female} <<<==============")
        acc = acc_blonde_female

    if print_non_blonde:
        non_blonde_df = df[df["out_put_GT"] == 0]
        non_blonde_male_df = non_blonde_df[non_blonde_df[attribute_col] == 1]
        non_blonde_female_df = non_blonde_df[non_blonde_df[attribute_col] == 0]
        print(f"GT Shape non blonde {concept_1}: {non_blonde_male_df.shape}")
        print(
            f"Pred Shape non blonde {concept_1}: {non_blonde_male_df[non_blonde_male_df[neg_pred_col] == 0].shape}"
        )
        acc_non_blonde_male = non_blonde_male_df[non_blonde_male_df[neg_pred_col] == 0].shape[0] / \
                              non_blonde_male_df.shape[0]
        print(f"acc non blonde {concept_1}: {acc_non_blonde_male}")

        print(f"GT Shape non blonde {concept_2}: {non_blonde_female_df.shape}")
        print(
            f"Pred Shape non blonde {concept_2}: {non_blonde_female_df[non_blonde_female_df[neg_pred_col] == 0].shape}"
        )
        acc_non_blonde_female = non_blonde_female_df[non_blonde_female_df[neg_pred_col] == 0].shape[0] / \
                                non_blonde_female_df.shape[0]

        print(f"acc non blonde {concept_2}: {acc_non_blonde_female}")
        print("#############" * 10)

    if log_file:
        with open(log_file, 'w') as f:
            print(f"Avg acc: {avg}", file=f)
            print(f"Worst Group acc: {acc_blonde_male}", file=f)

    return acc


def calculate_worst_group_acc_rsna_mammo(df, pred_col, attribute_col="calc"):
    df["out_put_predict_bin"] = (df["out_put_predict"] >= 0.5).astype(int)
    aucroc = auroc(gt=df["out_put_GT"].values, pred=df["out_put_predict"].values)
    print("Overall AUC-ROC for the whole dataset (Initial model): ", aucroc)
    if f"{pred_col}_proba" in df.columns:
        aucroc = auroc(gt=df["out_put_GT"].values, pred=df[f"{pred_col}_proba"].values)
        print("Overall AUC-ROC for the whole dataset (Cur model): ", aucroc)
    print(f"df: {df.shape}")

    if "_bin" not in pred_col:
        pred_col = f"{pred_col}_bin"
    print(f"################################# RSNA Cancer with {attribute_col} #################################")
    cancer_df = df[df["out_put_GT"] == 1]
    cancer_df_w_attr = cancer_df[cancer_df[attribute_col] == 1]
    cancer_df_wo_attr = cancer_df[cancer_df[attribute_col] == 0]
    print(f"[Shape] Cancer GT with {attribute_col}: {cancer_df_w_attr.shape}")
    print(
        f"[Shape] Cancer Pred with {attribute_col}: {cancer_df_w_attr[cancer_df_w_attr[pred_col] == 1].shape}"
    )
    acc_cancer_attr = cancer_df_w_attr[cancer_df_w_attr[pred_col] == 1].shape[0] / cancer_df_w_attr.shape[0]

    print(f"[Shape] Cancer GT without {attribute_col}: {cancer_df_wo_attr.shape}")
    print(
        f"[Shape] Cancer Pred without {attribute_col}: {cancer_df_wo_attr[cancer_df_wo_attr[pred_col] == 1].shape}"
    )
    acc_cancer_wo_attr = cancer_df_wo_attr[cancer_df_wo_attr[pred_col] == 1].shape[0] / cancer_df_wo_attr.shape[0]

    print(f"acc Cancer with {attribute_col}: {acc_cancer_attr}")
    print(f"acc Cancer without {attribute_col}: {acc_cancer_wo_attr}")

    print(f"Mean accuracy: {df[df[pred_col] == df['out_put_GT']].shape[0] / df.shape[0]}")

    return acc_cancer_wo_attr


def calculate_worst_group_acc_med_img(
        df, pos_pred_col, neg_pred_col, attribute_col, log_file=None, disease="Pneumothorax"):
    print(f"Dataset shape: {df.shape}")
    df[f"{pos_pred_col}_bin"] = (df[pos_pred_col] >= 0.5).astype(int)
    df_pt = df[df["out_put_GT"] == 1]
    df_pt_with_tube = df_pt[df_pt[attribute_col] == 1]
    df_pt_without_tube = df_pt[df_pt[attribute_col] == 0]
    print(f"{disease} patients:", df_pt.shape)
    print(f"{disease} patients with {attribute_col}:", df_pt_with_tube.shape)
    print(f"{disease} patients without {attribute_col}:", df_pt_without_tube.shape)
    accuracy_with_tube = (df_pt_with_tube['out_put_GT'] == df_pt_with_tube[f"{pos_pred_col}_bin"]).mean()
    accuracy_without_tube = (df_pt_without_tube['out_put_GT'] == df_pt_without_tube[f"{pos_pred_col}_bin"]).mean()
    accuracy_overall = (df_pt['out_put_GT'] == df_pt[f"{pos_pred_col}_bin"]).mean()

    print(f"Accuracy for {disease} patients without {attribute_col} (error slice) after mitigation:",
          accuracy_without_tube)
    print(f"Accuracy for {disease} overall patients:", accuracy_overall)

    df_positives_with_tube = df[(df['out_put_GT'] == 1) & (df[attribute_col] == 1)]
    gt_with_tube = df_positives_with_tube["out_put_GT"]
    pred_with_tube = df_positives_with_tube[pos_pred_col]

    df_positives_without_tube = df[(df['out_put_GT'] == 1) & (df[attribute_col] == 0)]
    gt_without_tube = df_positives_without_tube["out_put_GT"]
    pred_without_tube = df_positives_without_tube[pos_pred_col]

    df_negatives = df[df['out_put_GT'] == 0]
    gt_negatives = df_negatives["out_put_GT"]
    pred_negatives = df_negatives[neg_pred_col]

    tot_gt_with_tube = np.concatenate((gt_with_tube, gt_negatives), axis=0)
    tot_gt_without_tube = np.concatenate((gt_without_tube, gt_negatives), axis=0)
    tot_pred_with_tube = np.concatenate((pred_with_tube, pred_negatives), axis=0)
    tot_pred_without_tube = np.concatenate((pred_without_tube, pred_negatives), axis=0)

    tot_gt = np.concatenate((gt_with_tube, gt_without_tube, gt_negatives), axis=0)
    tot_pred = np.concatenate((pred_with_tube, pred_without_tube, pred_negatives), axis=0)

    auroc = roc_auc_score(tot_gt, tot_pred)
    auroc_with_tube = roc_auc_score(tot_gt_with_tube, tot_pred_with_tube)
    auroc_without_tube = roc_auc_score(tot_gt_without_tube, tot_pred_without_tube)

    print("AUROC for overall:", auroc)
    print(f"AUROC for positives disease with {attribute_col} vs all negatives:", auroc_with_tube)
    print(f"AUROC for positives disease without {attribute_col} vs all negatives:", auroc_without_tube)

    if log_file:
        with open(log_file, 'w') as f:
            print(f"Accuracy for {disease} patients with {attribute_col}: {accuracy_with_tube}", file=f)
            print(f"Accuracy for {disease} patients without  {attribute_col} (Worst Group): {accuracy_without_tube}",
                  file=f)
            print(f"Accuracy for {disease} overall patients: {accuracy_overall}", file=f)
            print("\n", file=f)
            print(f"AUROC for overall (Mean):", auroc, file=f)
            print(f"AUROC for positive disease with  {attribute_col} vs all negatives: {auroc_with_tube}", file=f)
            print(f"AUROC for positive disease without  {attribute_col} vs all negatives: {auroc_without_tube}", file=f)

    return accuracy_without_tube


def calculate_worst_group_acc_chexpert_no_findings(df, pred_col, attribute_col="attribute"):
    df[pred_col] = (df[pred_col] >= 0.5).astype(int)
    print(f"df: {df.shape}")
    print(df.head())

    print("#############" * 10)
    print("####################################### Waterbirds Stats #######################################")
    no_findings = df[df["out_put_GT"] == 1]
    no_findings_male_white = no_findings[no_findings[attribute_col] == 0]
    no_findings_male_black = no_findings[no_findings[attribute_col] == 1]
    no_findings_male_other = no_findings[no_findings[attribute_col] == 2]
    no_findings_female_white = no_findings[no_findings[attribute_col] == 3]
    no_findings_female_black = no_findings[no_findings[attribute_col] == 4]
    no_findings_female_other = no_findings[no_findings[attribute_col] == 5]

    print(f"GT Shape no_findings_male_white: {no_findings_male_white.shape}")
    print(
        f"Pred Shape no_findings_male_white: {no_findings_male_white[no_findings_male_white[pred_col] == 1].shape}"
    )
    acc_no_findings_male_white = no_findings_male_white[no_findings_male_white[pred_col] == 1].shape[0] / \
                                 no_findings_male_white.shape[0]
    print(f"acc acc_no_findings_male_white: {acc_no_findings_male_white}")

    print("============================================================")
    print(f"GT Shape no_findings_male_black: {no_findings_male_black.shape}")
    print(
        f"Pred Shape no_findings_male_black: {no_findings_male_black[no_findings_male_black[pred_col] == 1].shape}"
    )
    acc_no_findings_male_black = no_findings_male_black[no_findings_male_black[pred_col] == 1].shape[0] / \
                                 no_findings_male_black.shape[0]
    print(f"acc no_findings_male_black: {acc_no_findings_male_black}")

    print("============================================================")
    print(f"GT Shape no_findings_male_other: {no_findings_male_other.shape}")
    print(
        f"Pred Shape no_findings_male_other: {no_findings_male_other[no_findings_male_other[pred_col] == 1].shape}"
    )
    acc_no_findings_male_other = no_findings_male_other[no_findings_male_other[pred_col] == 1].shape[0] / \
                                 no_findings_male_other.shape[0]
    print(f"acc no_findings_male_other: {acc_no_findings_male_other}")


# def calculate_performance_metrics_urbancars_df(df, split, bg_ratio, co_occur_obj_ratio, label="out_put_predict"):
#     log_dict = {}
#
#     # Conditions
#     bg_absent = df['bg'] == 0
#     bg_present = df['bg'] == 1
#     co_occur_obj_absent = df['co-occur'] == 0
#     co_occur_obj_present = df['co-occur'] == 1
#
#     # Calculate accuracy based on conditions
#     def calculate_accuracy(condition, label=label):
#         correct = df.loc[condition, 'out_put_GT'] == df.loc[condition, label]
#         return correct.mean()
#
#     absent_present_str_list = ["absent", "present"]
#     absent_present_bg_ratio_list = [1 - bg_ratio, bg_ratio]
#     absent_present_co_occur_obj_ratio_list = [1 - co_occur_obj_ratio, co_occur_obj_ratio]
#
#     weighted_group_acc = 0
#     for bg_shortcut in range(len(absent_present_str_list)):
#         for second_shortcut in range(len(absent_present_str_list)):
#             if bg_shortcut == 0:
#                 first_condition = bg_absent
#                 bg_shortcut_str = "absent"
#             else:
#                 first_condition = bg_present
#                 bg_shortcut_str = "present"
#
#             if second_shortcut == 0:
#                 second_condition = co_occur_obj_absent
#                 co_occur_obj_shortcut_str = "absent"
#             else:
#                 second_condition = co_occur_obj_present
#                 co_occur_obj_shortcut_str = "present"
#
#             mask = first_condition & second_condition
#             acc = calculate_accuracy(mask, label=label)
#
#             log_dict[f"{split}_bg_{bg_shortcut_str}_co_occur_obj_{co_occur_obj_shortcut_str}_acc"] = acc
#
#             cur_group_bg_ratio = absent_present_bg_ratio_list[bg_shortcut]
#             cur_group_co_occur_obj_ratio = absent_present_co_occur_obj_ratio_list[second_shortcut]
#             cur_group_ratio = cur_group_bg_ratio * cur_group_co_occur_obj_ratio
#             weighted_group_acc += acc * cur_group_ratio
#
#     bg_gap = log_dict[f"{split}_bg_absent_co_occur_obj_present_acc"] - weighted_group_acc
#     co_occur_obj_gap = log_dict[f"{split}_bg_present_co_occur_obj_absent_acc"] - weighted_group_acc
#     both_gap = log_dict[f"{split}_bg_absent_co_occur_obj_absent_acc"] - weighted_group_acc
#
#     log_dict.update({
#         f"{split}_id_acc": weighted_group_acc,
#         f"{split}_bg_gap": bg_gap,
#         f"{split}_co_occur_obj_gap": co_occur_obj_gap,
#         f"{split}_both_gap": both_gap,
#     })
#
#     print(log_dict)
#     return log_dict

def calculate_performance_metrics_urbancars_df(clf, loader, split, device, bg_ratio=0.95, co_occur_obj_ratio=0.95):
    meter = MultiDimAverageMeter(
        (2, 2, 2)
    )
    total_correct = []
    total_bg_correct = []
    total_co_occur_obj_correct = []
    total_shortcut_conflict_mask = []

    pbar = tqdm(loader, dynamic_ncols=True)
    for data_dict in pbar:
        image, target = data_dict["image"], data_dict["label"]
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):
            reps_classifier = clf["feature_maps"](image)
            output = clf["classifier"](reps_classifier)

        pred = output.argmax(dim=1)

        obj_label = target[:, 0]
        bg_label = target[:, 1]
        co_occur_obj_label = target[:, 2]

        shortcut_conflict_mask = bg_label != co_occur_obj_label
        total_shortcut_conflict_mask.append(shortcut_conflict_mask.cpu())

        correct = pred == obj_label
        meter.add(correct.cpu(), target.cpu())
        total_correct.append(correct.cpu())

        bg_correct = pred == bg_label
        total_bg_correct.append(bg_correct.cpu())

        co_occur_obj_correct = pred == co_occur_obj_label
        total_co_occur_obj_correct.append(co_occur_obj_correct.cpu())

    num_correct = meter.cum.reshape(*meter.dims)
    cnt = meter.cnt.reshape(*meter.dims)
    multi_dim_color_acc = num_correct / cnt
    log_dict = {}
    absent_present_str_list = ["absent", "present"]
    absent_present_bg_ratio_list = [1 - bg_ratio, bg_ratio]
    absent_present_co_occur_obj_ratio_list = [1 - co_occur_obj_ratio, co_occur_obj_ratio]

    weighted_group_acc = 0
    for bg_shortcut in range(len(absent_present_str_list)):
        for second_shortcut in range(len(absent_present_str_list)):
            first_shortcut_mask = (meter.eye_tsr == bg_shortcut).unsqueeze(2)
            co_occur_obj_shortcut_mask = (
                    meter.eye_tsr == second_shortcut
            ).unsqueeze(1)
            mask = first_shortcut_mask * co_occur_obj_shortcut_mask
            acc = multi_dim_color_acc[mask].mean().item()
            bg_shortcut_str = absent_present_str_list[bg_shortcut]
            co_occur_obj_shortcut_str = absent_present_str_list[
                second_shortcut
            ]
            log_dict[
                f"{split}_bg_{bg_shortcut_str}"
                f"_co_occur_obj_{co_occur_obj_shortcut_str}_acc"
            ] = acc
            cur_group_bg_ratio = absent_present_bg_ratio_list[bg_shortcut]
            cur_group_co_occur_obj_ratio = (
                absent_present_co_occur_obj_ratio_list[second_shortcut]
            )
            cur_group_ratio = (
                    cur_group_bg_ratio * cur_group_co_occur_obj_ratio
            )
            weighted_group_acc += acc * cur_group_ratio

    bg_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_present_acc"]
            - weighted_group_acc
    )
    co_occur_obj_gap = (
            log_dict[f"{split}_bg_present_co_occur_obj_absent_acc"]
            - weighted_group_acc
    )
    both_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_absent_acc"]
            - weighted_group_acc
    )

    log_dict.update(
        {
            f"{split}_id_acc": weighted_group_acc,
            f"{split}_bg_gap": bg_gap,
            f"{split}_co_occur_obj_gap": co_occur_obj_gap,
            f"{split}_both_gap": both_gap,
        }
    )

    total_bg_correct = torch.cat(total_bg_correct, dim=0)
    total_co_occur_obj_correct = torch.cat(
        total_co_occur_obj_correct, dim=0
    )
    total_correct = torch.cat(total_correct, dim=0)

    (
        bg_worst_group_acc,
        co_occur_obj_worst_group_acc,
        both_worst_group_acc,
    ) = meter.get_worst_group_acc()

    log_dict.update(
        {
            f"{split}_bg_worst_group_acc": bg_worst_group_acc,
            f"{split}_co_occur_obj_worst_group_acc": co_occur_obj_worst_group_acc,
            f"{split}_both_worst_group_acc": both_worst_group_acc,
        }
    )
    obj_acc = total_correct.float().mean().item()
    bg_acc = total_bg_correct.float().mean().item()
    co_occur_obj_acc = total_co_occur_obj_correct.float().mean().item()

    log_dict.update(
        {
            f"{split}_cue_obj_acc": obj_acc,
            f"{split}_cue_bg_acc": bg_acc,
            f"{split}_cue_co_occur_obj_acc": co_occur_obj_acc,
        }
    )

    print(log_dict)

    return log_dict


def calculate_performance_metrics_urbancars_df_ensemble(
        df, prediction_col, split, bg_ratio=0.95, co_occur_obj_ratio=0.95, log_file=None):
    print("################ Computing Whac A Mole Metric ################")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    data_dir = "/restricted/projectnb/batmanlab/rsyed/car/data/"
    test_set = UrbanCars(data_dir, "test", transform=test_transform)

    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(len(test_set), len(loader))
    print(f"df: {df.shape}")
    meter = MultiDimAverageMeter(
        (2, 2, 2)
    )
    total_correct = []
    total_bg_correct = []
    total_co_occur_obj_correct = []
    total_shortcut_conflict_mask = []

    pbar = tqdm(loader, dynamic_ncols=True)
    for idx, data_dict in enumerate(pbar):
        image, target = data_dict["image"], data_dict["label"]
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Get the output from the DataFrame
        output = torch.tensor(df.iloc[idx][prediction_col], device=device)
        pred = output

        obj_label = target[:, 0]
        bg_label = target[:, 1]
        co_occur_obj_label = target[:, 2]

        shortcut_conflict_mask = bg_label != co_occur_obj_label
        total_shortcut_conflict_mask.append(shortcut_conflict_mask.cpu())

        correct = pred == obj_label
        meter.add(correct.cpu(), target.cpu())
        total_correct.append(correct.cpu())

        bg_correct = pred == bg_label
        total_bg_correct.append(bg_correct.cpu())

        co_occur_obj_correct = pred == co_occur_obj_label
        total_co_occur_obj_correct.append(co_occur_obj_correct.cpu())

    num_correct = meter.cum.reshape(*meter.dims)
    cnt = meter.cnt.reshape(*meter.dims)
    multi_dim_color_acc = num_correct / cnt
    log_dict = {}
    absent_present_str_list = ["absent", "present"]
    absent_present_bg_ratio_list = [1 - bg_ratio, bg_ratio]
    absent_present_co_occur_obj_ratio_list = [1 - co_occur_obj_ratio, co_occur_obj_ratio]

    weighted_group_acc = 0
    for bg_shortcut in range(len(absent_present_str_list)):
        for second_shortcut in range(len(absent_present_str_list)):
            first_shortcut_mask = (meter.eye_tsr == bg_shortcut).unsqueeze(2)
            co_occur_obj_shortcut_mask = (
                    meter.eye_tsr == second_shortcut
            ).unsqueeze(1)
            mask = first_shortcut_mask * co_occur_obj_shortcut_mask
            acc = multi_dim_color_acc[mask].mean().item()
            bg_shortcut_str = absent_present_str_list[bg_shortcut]
            co_occur_obj_shortcut_str = absent_present_str_list[
                second_shortcut
            ]
            log_dict[
                f"{split}_bg_{bg_shortcut_str}"
                f"_co_occur_obj_{co_occur_obj_shortcut_str}_acc"
            ] = acc
            cur_group_bg_ratio = absent_present_bg_ratio_list[bg_shortcut]
            cur_group_co_occur_obj_ratio = (
                absent_present_co_occur_obj_ratio_list[second_shortcut]
            )
            cur_group_ratio = (
                    cur_group_bg_ratio * cur_group_co_occur_obj_ratio
            )
            weighted_group_acc += acc * cur_group_ratio

    bg_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_present_acc"]
            - weighted_group_acc
    )
    co_occur_obj_gap = (
            log_dict[f"{split}_bg_present_co_occur_obj_absent_acc"]
            - weighted_group_acc
    )
    both_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_absent_acc"]
            - weighted_group_acc
    )

    log_dict.update(
        {
            f"{split}_id_acc": weighted_group_acc,
            f"{split}_bg_gap": bg_gap,
            f"{split}_co_occur_obj_gap": co_occur_obj_gap,
            f"{split}_both_gap": both_gap,
        }
    )

    total_bg_correct = torch.cat(total_bg_correct, dim=0)
    total_co_occur_obj_correct = torch.cat(
        total_co_occur_obj_correct, dim=0
    )
    total_correct = torch.cat(total_correct, dim=0)

    (
        bg_worst_group_acc,
        co_occur_obj_worst_group_acc,
        both_worst_group_acc,
    ) = meter.get_worst_group_acc()

    log_dict.update(
        {
            f"{split}_bg_worst_group_acc": bg_worst_group_acc,
            f"{split}_co_occur_obj_worst_group_acc": co_occur_obj_worst_group_acc,
            f"{split}_both_worst_group_acc": both_worst_group_acc,
        }
    )
    obj_acc = total_correct.float().mean().item()
    bg_acc = total_bg_correct.float().mean().item()
    co_occur_obj_acc = total_co_occur_obj_correct.float().mean().item()

    log_dict.update(
        {
            f"{split}_cue_obj_acc": obj_acc,
            f"{split}_cue_bg_acc": bg_acc,
            f"{split}_cue_co_occur_obj_acc": co_occur_obj_acc,
        }
    )

    print(log_dict)

    with open(log_file, 'w') as file:
        for key, value in log_dict.items():
            file.write(f"{key}: {value}\n")

    return log_dict
