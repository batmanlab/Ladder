import netcal.metrics
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score, brier_score_loss, log_loss, classification_report,
                             precision_recall_curve, auc)

import torch
class MultiDimAverageMeter:
    # reference: https://github.com/alinlab/LfF/blob/master/util.py

    def __init__(self, dims):
        self.dims = dims
        self.eye_tsr = torch.eye(dims[0]).long()
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_worst_group_acc(self):
        num_correct = self.cum.reshape(*self.dims)
        cnt = self.cnt.reshape(*self.dims)

        first_shortcut_worst_group_acc = (
                num_correct.sum(dim=2) / cnt.sum(dim=2)
        ).min()
        second_shortcut_worst_group_acc = (
                num_correct.sum(dim=1) / cnt.sum(dim=1)
        ).min()
        both_worst_group_acc = (num_correct / cnt).min()

        return (
            first_shortcut_worst_group_acc,
            second_shortcut_worst_group_acc,
            both_worst_group_acc,
        )


def eval_metrics_vision(targets, attributes, preds, gs):
    """
    Evaluate metrics for vision tasks
    :param targets:
    :param attributes:
    :param preds:
    :param gs:

    Returns
    - binary_metrics: Binary classification metrics
    """
    thres = 0.5
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    label_set = np.unique(targets)

    res = {}
    res['overall'] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set)
    }
    res['per_attribute'] = {}
    res['per_class'] = {}
    res['per_group'] = {}

    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][int(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set)
        }

    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
    res['overall']['macro_avg'] = classes_report['macro avg']
    res['overall']['weighted_avg'] = classes_report['weighted avg']
    for y in np.unique(targets):
        res['per_class'][int(y)] = classes_report[str(y)]

    for g in np.unique(gs):
        mask = gs == g
        res['per_group'][g] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set)
        }

    res['adjusted_accuracy'] = sum([res['per_group'][g]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
    res['min_attr'] = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr'] = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
    res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()

    return res


def binary_metrics(targets, preds, label_set=[0, 1], return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item()
        res['FN'] = CM[1][0].item()
        res['TP'] = CM[1][1].item()
        res['FP'] = CM[0][1].item()
        res['error'] = res['FN'] + res['FP']

        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP'] / (res['TP'] + res['FN'])
            res['FNR'] = res['FN'] / (res['TP'] + res['FN'])

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP'] / (res['FP'] + res['TN'])
            res['TNR'] = res['TN'] / (res['FP'] + res['TN'])

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
    else:
        CM = confusion_matrix(targets, preds, labels=label_set)
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res


def prob_metrics(targets, preds, label_set, return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'AUROC_ovo': roc_auc_score(targets, preds, multi_class='ovo', labels=label_set),
        'BCE': log_loss(targets, preds, eps=1e-6, labels=label_set),
        'ECE': netcal.metrics.ECE().measure(preds, targets)
    }

    # happens when you predict a class, but there are no samples with that class in the dataset
    try:
        res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set)
    except:
        res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovo', labels=label_set)

    if len(set(targets)) == 2:
        res['AUPRC'] = average_precision_score(targets, preds, average='macro')
        res['brier'] = brier_score_loss(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res


def pr_auc(gt, pred, get_all=False):
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score


def pfbeta_binarized(gt, pred):
    positives = pred[gt == 1]
    scores = []
    for th in positives:
        binarized = (pred >= th).astype('int')
        score = pfbeta(gt, binarized, 1)
        scores.append(score)

    return np.max(scores)


def auroc(gt, pred):
    return roc_auc_score(gt, pred)


def compute_auprc(gt, pred):
    return average_precision_score(gt, pred)


def compute_accuracy_np_array(gt, pred):
    return np.mean(gt == pred)


def pfbeta(gt, pred, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gt)):
        prediction = min(max(pred[idx], 0), 1)
        if (gt[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def pr_auc(gt, pred, get_all=False):
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score
