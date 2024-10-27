import numpy as np
from sklearn.metrics import auc, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs


def compute_accuracy(gt, pred):
    return (((pred == gt).sum()) / gt.size(0)).item() * 100


def compute_auprc(gt, pred):
    return average_precision_score(gt, pred)


def compute_accuracy_np_array(gt, pred):
    return np.mean(gt == pred)


def auroc(gt, pred):
    return roc_auc_score(gt, pred)
