import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device(args):
    return "cuda" if args.device == "cuda" else 'cpu'


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_Paths(args):
    chk_pt_path = Path(f"{args.checkpoints}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")
    output_path = Path(f"{args.output_path}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")
    tb_logs_path = Path(f"{args.tensorboard_path}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")

    return chk_pt_path, output_path, tb_logs_path


def stratified_sample(df, n, label):
    df_0 = df[df[label] == 0]
    df_1 = df[df[label] == 1]

    # Sample n/2 from each class
    df_0_sampled = df_0.sample(n=n // 2, random_state=42)
    df_1_sampled = df_1.sample(n=n // 2, random_state=42)

    return pd.concat([df_0_sampled, df_1_sampled])


def get_constant_prompts(prompt_dict):
    _prompts = {}
    for k in prompt_dict:
        _prompts[k] = [prompt_dict[k]]

    return _prompts


def process_class_prompts(cls_prompts):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer.model_max_length = 77
    cls_prompt_inputs = defaultdict()
    for k, v in cls_prompts.items():
        text_inputs = tokenizer(v, truncation=True, padding=True, return_tensors='pt')
        cls_prompt_inputs[k] = text_inputs
    return cls_prompt_inputs


def _split_report_into_segment(report):
    """clean up raw reports into sentences"""
    if pd.isnull(report):
        return []
    else:
        report = report.replace("\n", " ")
        # splitter = re.compile("[0-9]+\.")
        splitter = re.compile("[0-9]+\.+[^0-9]")
        report = splitter.split(report)
        reports = [point.split(". ") for point in report]
        # reports = [point.split(".") for point in report]
        reports = [sent for point in reports for sent in point]
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            # tokenizer = RegexpTokenizer(r"\w+")
            # tokens = tokenizer.tokenize(sent.lower())

            tokens = nltk.wordpunct_tokenize(sent.lower())

            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 4:  # only include relative long sentences
                study_sent.append(" ".join(included_tokens))
        return study_sent


def _split_report_into_segment_breast(report):
    """clean up raw reports into sentences"""
    if pd.isnull(report):
        return []
    else:
        report = report.replace("\n", " ")
        # splitter = re.compile("[0-9]+\.")
        splitter = re.compile("[0-9]+\.+[^0-9]")
        report = splitter.split(report)
        reports = [point.split(". ") for point in report]
        # reports = [point.split(".") for point in report]
        reports = [sent for point in reports for sent in point]
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            # tokenizer = RegexpTokenizer(r"\w+")
            # tokens = tokenizer.tokenize(sent.lower())

            tokens = nltk.wordpunct_tokenize(sent.lower())

            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 3:  # only include relative long sentences
                study_sent.append(" ".join(included_tokens))
        return study_sent


def _split_report_into_segment_nih(report):
    """clean up raw reports into sentences"""
    if pd.isnull(report):
        return []
    else:
        report = report.replace("\n", " ")
        # splitter = re.compile("[0-9]+\.")
        splitter = re.compile("[0-9]+\.+[^0-9]")
        report = splitter.split(report)
        reports = [point.split(". ") for point in report]
        # reports = [point.split(".") for point in report]
        reports = [sent for point in reports for sent in point]
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            # tokenizer = RegexpTokenizer(r"\w+")
            # tokens = tokenizer.tokenize(sent.lower())

            tokens = nltk.wordpunct_tokenize(sent.lower())

            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 4:  # only include relative long sentences
                study_sent.append(" ".join(included_tokens))
        return study_sent


def _split_report_into_segment_concat_nih(report):
    """clean up raw reports into sentences"""
    if pd.isnull(report):
        return []
    else:
        report = report.replace("\n", " ")
        # splitter = re.compile("[0-9]+\.")
        splitter = re.compile("[0-9]+\.+[^0-9]")
        report = splitter.split(report)
        reports = [point.split(". ") for point in report]
        # reports = [point.split(".") for point in report]
        reports = [sent for point in reports for sent in point]
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            # tokenizer = RegexpTokenizer(r"\w+")
            # tokens = tokenizer.tokenize(sent.lower())

            tokens = nltk.wordpunct_tokenize(sent.lower())

            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            study_sent.append(" ".join(included_tokens))
        concatenated_string = ""

        for sentence in study_sent:
            # Check if the sentence ends with a period
            if not sentence.endswith('.'):
                sentence += '.'

            # Concatenate the sentence to the result string
            concatenated_string += sentence.strip() + ' '
        return concatenated_string.strip()


def _split_report_into_segment_concat(report):
    """clean up raw reports into sentences"""
    if pd.isnull(report):
        return []
    else:
        report = report.replace("\n", " ")
        # splitter = re.compile("[0-9]+\.")
        splitter = re.compile("[0-9]+\.+[^0-9]")
        report = splitter.split(report)
        reports = [point.split(". ") for point in report]
        # reports = [point.split(".") for point in report]
        reports = [sent for point in reports for sent in point]
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            # tokenizer = RegexpTokenizer(r"\w+")
            # tokens = tokenizer.tokenize(sent.lower())

            tokens = nltk.wordpunct_tokenize(sent.lower())

            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            study_sent.append(" ".join(included_tokens))
        concatenated_string = ""

        for sentence in study_sent:
            # Check if the sentence ends with a period
            if not sentence.endswith('.'):
                sentence += '.'

            # Concatenate the sentence to the result string
            concatenated_string += sentence.strip() + ' '
        return concatenated_string.strip()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def get_input_shape(dataset):
    if (
            dataset.lower() == "waterbirds" or dataset.lower() == "celeba" or dataset.lower() == "nih" or
            dataset.lower() == "metashift" or dataset.lower() == "chexpertnofinding" or dataset.lower() == "urbancars"
            or dataset.lower() == "cub_200_2011_3d_toy_shape"
    ):
        input_shape = (3, 224, 224,)
        return input_shape


def get_hparams(dataset, arch):
    if dataset.lower() == "waterbirds" and arch.lower() == "resnet_sup_in1k":
        return {
            'resnet18': False,
            'nonlinear_classifier': False,
            'group_balanced': False,
            'pretrained': True,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'sgd',
            'last_layer_dropout': 0.0,
            'batch_size': 108,
            'image_arch': 'resnet_sup_in1k',
            'text_arch': 'bert-base-uncased',
            'steps': 5001
        }
    elif dataset.lower() == "waterbirds" and arch.lower() == "vit_sup_in1k":
        return {
            'resnet18': False,
            'nonlinear_classifier': False,
            'group_balanced': False,
            'pretrained': True,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'sgd',
            'last_layer_dropout': 0.0,
            'batch_size': 108,
            'image_arch': 'vit_sup_in1k',
            'text_arch': 'bert-base-uncased',
            'steps': 5001
        }
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "resnet_barlow_in1k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'resnet_barlow_in1k', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "resnet_dino_in1k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'resnet_dino_in1k', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "resnet_simclr_in1k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'resnet_simclr_in1k', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "resnet_sup_in21k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'resnet_sup_in21k', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "vit_clip_laion":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'vit_clip_laion', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "vit_clip_oai":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'vit_clip_oai', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "vit_dino_in1k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'vit_dino_in1k', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "vit_sup_in21k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'vit_sup_in21k', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif (
            dataset.lower() == "waterbirds" or 
            dataset.lower() == "nih" or dataset.lower() == "celeba" or dataset.lower() == "metashift"
    ) and arch.lower() == "vit_sup_swag":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False, 'pretrained': True,
                'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'vit_sup_swag', 'text_arch': 'bert-base-uncased', 'steps': 5001}
    elif dataset.lower() == "nih" and arch.lower() == "vit_sup_in1k":
        return {'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': True, 'pretrained': True,
                'lr': 1e-05, 'weight_decay': 0.0001, 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
                'image_arch': 'vit_sup_in1k', 'text_arch': 'bert-base-uncased', 'steps': 20001}
    elif dataset.lower() == "celeba" and arch.lower() == "resnet_sup_in1k":
        return {
            'resnet18': False,
            'nonlinear_classifier': False,
            'group_balanced': False,
            'pretrained': True,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'sgd',
            'last_layer_dropout': 0.0,
            'batch_size': 108,
            'image_arch': 'resnet_sup_in1k',
            'text_arch': 'bert-base-uncased',
            'steps': 30001
        }
    elif dataset.lower() == "celeba" and arch.lower() == "vit_sup_in1k":
        return {
            'resnet18': False,
            'nonlinear_classifier': False,
            'group_balanced': False,
            'pretrained': True,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'sgd',
            'last_layer_dropout': 0.0,
            'batch_size': 108,
            'image_arch': 'vit_sup_in1k',
            'text_arch': 'bert-base-uncased',
            'steps': 30001
        }
    elif dataset.lower() == "metashift" and arch.lower() == "resnet_sup_in1k":
        return {
            'resnet18': False,
            'nonlinear_classifier': False,
            'group_balanced': False,
            'pretrained': True,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'sgd',
            'last_layer_dropout': 0.0,
            'batch_size': 108,
            'image_arch': 'resnet_sup_in1k',
            'text_arch': 'bert-base-uncased', 'steps': 5001
        }
    elif dataset.lower() == "metashift" and arch.lower() == "vit_sup_in1k":
        return {
            'resnet18': False,
            'nonlinear_classifier': False,
            'group_balanced': False,
            'pretrained': True,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'sgd',
            'last_layer_dropout': 0.0,
            'batch_size': 108,
            'image_arch': 'vit_sup_in1k',
            'text_arch': 'bert-base-uncased',
            'steps': 5001
        }


# Define Binary Cross-Entropy Loss function
def binary_cross_entropy(y_true, y_pred):
    # Small constant to prevent errors due to log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return bce
