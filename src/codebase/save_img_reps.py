import copy
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset_factory import create_dataloaders
from metrics_factory import eval_metrics_vision, pfbeta_binarized, pr_auc, auroc, compute_auprc, \
    compute_accuracy_np_array
from metrics_factory.calculate_worst_group_acc import calculate_performance_metrics_urbancars_df
from model_factory import create_classifier, create_clip
from utils import seed_all, get_input_shape

warnings.filterwarnings("ignore")
import argparse
import os

os.environ['TRANSFORMERS_CACHE'] = '/restricted/projectnb/batmanlab/shawn24/.cache/huggingface/transformers'
os.environ['TORCH_HOME'] = '/restricted/projectnb/batmanlab/shawn24/.cache/torch'


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NIH", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--classifier", default="resnet_sup_in1k", type=str, help="architecture of the classifier")
    parser.add_argument(
        "--classifier_check_pt", metavar="DIR",
        default="./out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl",
        help=""
    )
    parser.add_argument("--clip_check_pt", metavar="DIR", default="", help="")
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}",
        help=""
    )
    parser.add_argument(
        "--flattening-type", default="adaptive", type=str,
        help="adaptive or flattened (Adaptive avg pooled features or flattened features?)"
    )
    parser.add_argument(
        "--clip_vision_encoder", default="RN50", type=str, help="vision encoder of medclip (Resnet50 or ViT)"
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default="0", type=int)
    parser.add_argument(
        "--tokenizers", default="", type=str, help="tokenizer path required by CXR-CLIP and Mammo-CLIP")
    parser.add_argument(
        "--cache_dir", default="", type=str, help="cache_dir required by CXR-CLIP and Mammo-CLIP")
    return parser.parse_args()


def save_additional_info_to_csv(additional_info, save_path, mode):
    """
    Save the additional information to a CSV file.

    Parameters:
    - additional_info: The additional information dictionary.
    - save_path: The path where to save the CSV file.
    - mode: The current mode (train, valid, test) to name the file appropriately.
    """
    additional_info_copied = copy.deepcopy(additional_info)
    for key, value in additional_info_copied.items():
        if isinstance(value, torch.Tensor):
            additional_info_copied[key] = value.numpy()
        elif isinstance(value, list):
            additional_info_copied[key] = np.concatenate(value)

    df = pd.DataFrame(additional_info_copied
                      )

    csv_file_path = save_path / f"{mode}_additional_info.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Saved additional info to {csv_file_path}")


def compute_performance_metrics(dataset, additional_info, save_path, mode):
    """
    Compute performance metrics for the given dataset.
    :param dataset:
    :param additional_info:
    :param save_path:

    Returns:
    - None
    """
    if dataset.lower() == 'waterbirds':
        targets = additional_info["out_put_GT"].numpy()
        preds = additional_info["out_put_predict"].numpy()
        attributes = additional_info["attribute_bg_predict"].numpy()
        gs = np.concatenate(additional_info["gs"])

        print(f"targets: {targets.shape}, preds: {preds.shape}, attributes: {attributes.shape}, gs: {gs.shape}")

        final_results = eval_metrics_vision(targets, attributes, preds, gs)

        pickle.dump(final_results, open(save_path / 'final_results.pkl', 'wb'))

        print(f"\n{mode} accuracy (best validation checkpoint):")
        print(f"\tmean:\t[{final_results['overall']['accuracy']:.3f}]\n"
              f"\tworst:\t[{final_results['min_group']['accuracy']:.3f}]")
        print("Group-wise accuracy:")
        print('\tgroup-wise {}'.format(
            (np.array2string(
                pd.DataFrame(final_results['per_group']).T['accuracy'].values,
                separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    elif dataset.lower() == 'rsna' or dataset.lower() == "vindr":
        for key, value in additional_info.items():
            if isinstance(value, torch.Tensor):
                additional_info[key] = value.numpy()
            elif isinstance(value, list):
                additional_info[key] = np.concatenate(value)

        df = pd.DataFrame(additional_info)
        oof_df_agg = df[['patient_id', 'laterality', "out_put_GT", 'out_put_predict']].groupby(
            ['patient_id', 'laterality']).mean()
        np_gt = oof_df_agg["out_put_GT"].values
        np_preds = oof_df_agg["out_put_predict"].values

        aucroc = auroc(np_gt, np_preds)

        mask = np_gt == 1
        np_gt_cancer = np_gt[mask]
        np_preds_cancer = np_preds[mask]
        np_preds_cancer = (np_preds_cancer >= 0.5).astype(int)
        acc_cancer = compute_accuracy_np_array(np_gt_cancer, np_preds_cancer)

        print(f"aucroc: {aucroc} acc_cancer: {acc_cancer}")

    elif dataset.lower() == 'nih':
        df = pd.DataFrame({
            'GT': additional_info["out_put_GT"].numpy(),
            'Predictions': additional_info["out_put_predict"].numpy(),
            'Predictions_bin': (additional_info["out_put_predict"].numpy() >= 0.5).astype(int),
            'Tubes': additional_info["tube"].numpy()
        })

        pneumothorax_df = df[df["GT"] == 1]
        overall_pneumothorax_acc = pneumothorax_df[pneumothorax_df["Predictions_bin"] == 1].shape[0] / \
                                   pneumothorax_df.shape[0]

        pneumothorax_w_tube_df = df[(df["GT"] == 1) & ((df["Tubes"] == 1))]
        pneumothorax_w_tube_acc = pneumothorax_w_tube_df[pneumothorax_w_tube_df["Predictions_bin"] == 1].shape[0] / \
                                  pneumothorax_w_tube_df.shape[0]

        pneumothorax_wo_tube_df = df[(df["GT"] == 1) & ((df["Tubes"] == 0))]
        pneumothorax_wo_tube_acc = pneumothorax_wo_tube_df[pneumothorax_wo_tube_df["Predictions_bin"] == 1].shape[0] / \
                                   pneumothorax_wo_tube_df.shape[0]

        AUROCs = roc_auc_score(df["GT"].values, df["Predictions"].values)
        corr = df[(df["GT"] == 1) & (df["Predictions_bin"] == 1)]
        incorr = df[(df["GT"] == 1) & (df["Predictions_bin"] == 0)]
        print(f"[Shape] corr: {corr.shape[0]}, incorr: {incorr.shape[0]}")
        print(f"[Shape] tube: {pneumothorax_w_tube_df.shape[0]}, no tube: {pneumothorax_wo_tube_df.shape[0]}")

        print(f"Overall AuROC: {AUROCs}")
        print(
            f"[Acc] overall pneumothorax: {overall_pneumothorax_acc}, "
            f"[Acc] pneumothorax with tube: {pneumothorax_w_tube_acc}, "
            f"[Acc] pneumothorax without tube: : {pneumothorax_wo_tube_acc}"
        )


def init_additional_info(dataset):
    """
    Initializes additional information to be saved.

    Parameters:
    - data_type: Type of data ('breast' or 'waterbirds').

    Returns:
    - additional_info: Dictionary to store additional information.
    """
    if (
            dataset.lower() == "waterbirds" or dataset.lower() == 'celeba'
            or dataset.lower() == 'metashift'
    ):
        # y=1 for waterbirds, y=0 for landbirds
        # a=1 for water background, y=0 for land background
        return {
            "out_put_GT": torch.FloatTensor(),
            "out_put_predict": torch.FloatTensor(),
            "attribute_bg_predict": torch.FloatTensor(),
            "idx": [],
            "gs": []
        }
    elif dataset.lower() == "rsna" or dataset.lower() == "vindr":
        return {
            "patient_id": torch.IntTensor(),
            "laterality": torch.IntTensor(),
            "out_put_GT": torch.FloatTensor(),
            "out_put_predict": torch.FloatTensor(),
            "mass": torch.FloatTensor(),
            "calc": torch.FloatTensor(),
            "mole": torch.FloatTensor(),
            "mark": torch.FloatTensor(),
            "scar": torch.FloatTensor(),
            "clip": torch.FloatTensor(),
        }
    elif dataset.lower() == "nih":
        return {
            "out_put_GT": torch.FloatTensor(),
            "out_put_predict": torch.FloatTensor(),
            "tube": torch.FloatTensor(),
            "effusion": torch.FloatTensor(),
            "img_path": []
        }


def update_additional_info(additional_info, batch_info, dataset):
    if (
            dataset.lower() == "waterbirds" or
            dataset.lower() == 'celeba' or
            dataset.lower() == 'metashift'
    ):
        idx, y, y_pred, bg_attr = batch_info["idx"], batch_info["y"], batch_info["y_pred"], batch_info["bg_attr"]
        additional_info["out_put_GT"] = torch.cat((additional_info["out_put_GT"], y), dim=0)
        additional_info["out_put_predict"] = torch.cat(
            (additional_info["out_put_predict"], y_pred), dim=0
        )
        additional_info["attribute_bg_predict"] = torch.cat(
            (additional_info["attribute_bg_predict"], bg_attr), dim=0
        )
        additional_info["gs"].append([f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(y, bg_attr))])
        additional_info["idx"].append(idx)
        return additional_info

    elif dataset.lower() == "nih":
        y, y_pred, tube, effusion, img_path = batch_info["y"], batch_info["y_pred"], batch_info["tube"], batch_info[
            "effusion"], batch_info["img_path"]
        additional_info["out_put_GT"] = torch.cat((additional_info["out_put_GT"], y), dim=0)
        additional_info["out_put_predict"] = torch.cat((additional_info["out_put_predict"], y_pred), dim=0)
        additional_info["tube"] = torch.cat((additional_info["tube"], tube), dim=0)
        additional_info["effusion"] = torch.cat((additional_info["effusion"], effusion), dim=0)
        additional_info["img_path"].append(img_path)
        return additional_info

    elif dataset.lower() == "rsna" or dataset.lower() == "vindr":
        y, y_pred, mass, calc, mole, mark, scar, clip, patient_id, laterality = batch_info["y"], batch_info["y_pred"], \
            batch_info["mass"], batch_info["calc"], \
            batch_info["mole"], batch_info["mark"], \
            batch_info["scar"], batch_info["clip"], \
            batch_info["patient_id"], \
            batch_info["laterality"]

        additional_info["patient_id"] = torch.cat((additional_info["patient_id"], patient_id), dim=0)
        additional_info["laterality"] = torch.cat((additional_info["laterality"], laterality), dim=0)
        additional_info["out_put_GT"] = torch.cat((additional_info["out_put_GT"], y), dim=0)
        additional_info["out_put_predict"] = torch.cat(
            (additional_info["out_put_predict"], y_pred), dim=0
        )
        additional_info["mass"] = torch.cat((additional_info["mass"], mass), dim=0)
        additional_info["calc"] = torch.cat((additional_info["calc"], calc), dim=0)
        additional_info["mole"] = torch.cat((additional_info["mole"], mole), dim=0)
        additional_info["mark"] = torch.cat((additional_info["mark"], mark), dim=0)
        additional_info["scar"] = torch.cat((additional_info["scar"], scar), dim=0)
        additional_info["clip"] = torch.cat((additional_info["clip"], clip), dim=0)
        return additional_info


def process_batch(batch, device, clf, clip_model, classifier_type, dataset):
    """
    Processes a batch of data to generate representations.
    :param batch:
    :param device:
    :param clf:
    :param clip:
    :param data_type:

    Returns:
    - reps_classifier: Representations from the classifier.
    - reps_clip: Representations from the CLIP model.
    - batch_info: Additional information for the batch.
    """
    if (
            dataset.lower() == 'waterbirds' or
            dataset.lower() == 'celeba' or
            dataset.lower() == 'metashift'
    ):
        i, img, y, bg_attr = batch
        img = img.to(device)
        reps_classifier = clf["feature_maps"](img)
        pred_logits = clf["classifier"](reps_classifier)

        y_pred = pred_logits.argmax(dim=1).detach().cpu()
        reps_clip = clip_model["model"].encode_image(img)
        reps_clip /= reps_clip.norm(dim=-1, keepdim=True)

        batch_info = {"idx": i, "y": y, "y_pred": y_pred, "bg_attr": bg_attr}
        return reps_classifier, reps_clip, batch_info

    elif dataset.lower() == "rsna" or dataset.lower() == "vindr":
        img = batch['img'].permute(0, 3, 1, 2).to(device)
        y = batch['label']
        mass = batch['mass']
        calc = batch['calc']
        clip = batch['clip']
        scar = batch['scar']
        mark = batch['mark']
        mole = batch['mole']
        patient_id = batch['patient_id']
        laterality = batch['laterality']

        reps_classifier, pred_logits = clf(img)
        y_pred = pred_logits.squeeze(1).sigmoid().to('cpu')
        reps_clip = clip_model["model"].encode_image_normalized(img)

        batch_info = {
            "y": y, "y_pred": y_pred, "mass": mass, "calc": calc, "clip": clip, "scar": scar, "mark": mark,
            "mole": mole, "patient_id": patient_id, "laterality": laterality
        }
        return reps_classifier, reps_clip, batch_info

    elif dataset.lower() == "nih" and clip_model["type"] == "cxr_clip" and classifier_type.lower() == "resnet50":
        img, y, tube, effusion, text = batch["img"], batch["label"], batch["tube"], batch["effusion"], batch["text"]
        img_path = batch["img_path"]
        img = img.to(device)
        pred_logits, _, reps_classifier = clf(img)
        y_pred = pred_logits.squeeze(1).sigmoid().to('cpu')

        reps_clip = clip_model["model"].encode_image(img.to(device))
        reps_clip = clip_model["model"].image_projection(reps_clip) if clip_model["model"].projection else reps_clip
        reps_clip = reps_clip / torch.norm(reps_clip, dim=1, keepdim=True)

        batch_info = {"y": y.squeeze(1), "y_pred": y_pred, "tube": tube, "effusion": effusion, "img_path": img_path}
        return reps_classifier, reps_clip, batch_info


def save_reps(loader, device, mode, clf, clip_model, save_path, classifier_type, dataset="breast"):
    """
    Processes and saves features and additional information for given data.

    Parameters:
    - loader: Data loader.
    - device: Computation device.
    - mode: Mode of operation (e.g., 'train', 'test').
    - clf: The classifier used for generating representations.
    - clip: The clip model for vision language representations.
    - save_path: Path to save the outputs.
    - dataset: Type of data ('breast' or 'waterbirds').
    - config: Additional configuration options.
    """

    all_reps_classifier = []
    all_reps_clip = []
    additional_info = init_additional_info(dataset)
    with torch.no_grad():
        with tqdm(total=len(loader), desc=f"Processing {mode} data") as t:
            for batch_id, batch in enumerate(loader):
                image_reps_clf, image_reps_clip, batch_info = process_batch(
                    batch, device, clf, clip_model, classifier_type, dataset)
                reps_classifier = [x.detach().cpu().numpy() for x in image_reps_clf]
                reps_clip = [x.detach().cpu().numpy() for x in image_reps_clip]
                all_reps_clip.extend(reps_clip)
                all_reps_classifier.extend(reps_classifier)
                additional_info = update_additional_info(additional_info, batch_info, dataset)

                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    all_reps_classifier = np.stack(all_reps_classifier)
    all_reps_clip = np.stack(all_reps_clip)
    print(
        f"Classifier {mode} embedding shape: {all_reps_classifier.shape}, "
        f"Clip {mode} embedding shape: {all_reps_clip.shape} "
    )
    np.save(save_path / f"{mode}_classifier_embeddings.npy", all_reps_classifier)
    np.save(save_path / f"{mode}_clip_embeddings.npy", all_reps_clip)

    if dataset.lower() == "urbancars":
        calculate_performance_metrics_urbancars_df(
            clf, loader, split=mode, device=device, bg_ratio=0.95, co_occur_obj_ratio=0.95)
    else:
        compute_performance_metrics(dataset, additional_info, save_path, mode=mode)

    with open(save_path / f"{mode}_additional_info.pkl", "wb") as f:
        pickle.dump(additional_info, f)
    save_additional_info_to_csv(additional_info, save_path, mode)

    print(f"Saved {mode} embeddings and additional information at {save_path}")


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.save_path = Path(args.save_path.format(args.seed)) / f"clip_img_encoder_{args.clip_vision_encoder}"
    args.save_path.mkdir(parents=True, exist_ok=True)
    print(args.save_path)
    args.input_shape = get_input_shape(args.dataset)
    clf = create_classifier(args)
    clip_model = create_clip(args)
    args.shuffle = False
    data_loaders = create_dataloaders(args)

    if "train_loader" in data_loaders:
        save_reps(data_loaders["train_loader"], args.device, "train", clf, clip_model, args.save_path, args.classifier,
                  args.dataset)
    if "test_loader" in data_loaders:
        save_reps(data_loaders["test_loader"], args.device, "test", clf, clip_model, args.save_path, args.classifier,
                  args.dataset)
    if "valid_loader" in data_loaders:
        save_reps(data_loaders["valid_loader"], args.device, "valid", clf, clip_model, args.save_path, args.classifier,
                  args.dataset)


if __name__ == "__main__":
    _args = config()
    main(_args)
