import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from metrics import auroc
from metrics_factory.calculate_worst_group_acc import calculate_worst_group_acc_waterbirds, \
    calculate_worst_group_acc_rsna_mammo, calculate_worst_group_acc_celebA, calculate_worst_group_acc_med_img, \
    calculate_worst_group_acc_metashift
from model_factory import create_classifier
from utils import seed_all, get_input_shape, AverageMeter
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import argparse
import os


def last_layer_retrain(
        model, epochs, train_data_loader, test_data_loader, criterion, optimizer, device,
        model_name, batch_size=32, loss_type="BCE"):
    model.to(device)
    best_accuracy = 0.0
    # model_new = None
    predictions_new = None
    for epoch in range(epochs):
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        losses = AverageMeter()
        progress_iter = tqdm(enumerate(train_data_loader), desc=f"[{epoch + 1:03d}/{epochs:03d} epoch train]",
                             total=len(train_data_loader))
        train_loss = 0
        for step, (x, y) in progress_iter:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                y_preds = model(x)
            if loss_type == "BCE":
                y = y.float().to(device)
                loss = criterion(y_preds.view(-1, 1), y.view(-1, 1))
            else:
                y = y.long().to(device)
                loss = criterion(y_preds, y)
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        print(f"[{epoch + 1:03d}/{epochs:03d} epoch train - Avg Loss: {losses.avg:.4f}]")
    torch.save({'model': model.state_dict()}, model_name)
    model.eval()
    progress_iter = tqdm(enumerate(test_data_loader), desc=f"[test]", total=len(test_data_loader))

    preds = []
    y_gt = []
    proba_list = []
    for step, (x, y) in progress_iter:
        x = x.to(device)
        y = y.float().to(device)
        with torch.no_grad():
            y_preds = model(x)
        if loss_type == "BCE":
            proba = y_preds.squeeze(1).sigmoid().to('cpu').numpy()
            proba_list.append(proba)
            preds.append((proba >= 0.5).astype(int))
        else:
            probabilities = F.softmax(y_preds, dim=1)
            proba = probabilities[:, 1].detach().cpu().numpy()
            proba_list.append(proba)
            preds.append(y_preds.argmax(dim=1).to('cpu').numpy())

        y_gt.append(y.to('cpu').numpy())
    binary_predictions = np.concatenate(preds)
    gt = np.concatenate(y_gt)
    accuracy = np.mean(binary_predictions == gt)

    if loss_type == "BCE":
        proba = np.concatenate(proba_list)
        np_gt = gt
        np_preds = proba
        aucroc = auroc(np_gt, np_preds)

        print("==================================== Overall Metrics =====================================")
        print(f"aucroc: {aucroc}")
        print(f"Test accuracy: {accuracy}")
        print("==================================== Overall Metrics =====================================")
    else:
        print(f"Test accuracy: {accuracy}")
    print(binary_predictions.shape)

    if loss_type == "BCE":
        return binary_predictions, np.concatenate(proba_list)
    else:
        return binary_predictions, np.concatenate(proba_list)


def generate_ds_last_layer_retrain(
        tr_df, test_df, clf_image_emb_path, batch_size, seed, n_samples,
        col_name_0="H3_chest_tubes_positioning", col_name_1="H3_chest_tubes_positioning"):
    img_emb_clf = np.load(clf_image_emb_path.format(seed, "valid"))
    print(tr_df.shape, img_emb_clf.shape)
    print(tr_df.columns)

    pt_df = tr_df[(tr_df["out_put_GT"] == 1)]
    print(pt_df.shape)
    pt_sorted_df = pt_df.sort_values(by=col_name_1, ascending=False)
    pt_top = pt_sorted_df.head(n_samples)
    pt_bottom = pt_sorted_df.tail(n_samples)

    no_pt_df = tr_df[(tr_df["out_put_GT"] == 0)]
    print(no_pt_df.shape)
    no_pt_sorted_df = no_pt_df.sort_values(by=col_name_0, ascending=False)
    no_pt_top = no_pt_sorted_df.head(n_samples)
    no_pt_bottom = pt_sorted_df.tail(n_samples)
    bal_df = pd.concat([pt_top, pt_bottom, no_pt_top, no_pt_bottom], axis=0)
    print(bal_df.shape)

    pt_top_img_emb_clf = torch.from_numpy(img_emb_clf[pt_top.index.tolist()])
    pt_bottom_img_emb_clf = torch.from_numpy(img_emb_clf[pt_bottom.index.tolist()])
    no_pt_top_img_emb_clf = torch.from_numpy(img_emb_clf[no_pt_top.index.tolist()])
    no_pt_bottom_img_emb_clf = torch.from_numpy(img_emb_clf[no_pt_bottom.index.tolist()])
    tr_img_emb_clf = torch.cat(
        [pt_top_img_emb_clf, pt_bottom_img_emb_clf, no_pt_top_img_emb_clf, no_pt_bottom_img_emb_clf], dim=0)
    # idx = bal_df.index.tolist()
    # tr_img_emb_clf = torch.from_numpy(img_emb_clf[idx])
    gt = torch.from_numpy(bal_df["out_put_GT"].values)
    train_dataset = TensorDataset(tr_img_emb_clf, gt)

    img_emb_clf = np.load(clf_image_emb_path.format(seed, "test"))
    image_emb = torch.from_numpy(img_emb_clf)
    gt = torch.from_numpy(test_df["out_put_GT"].values)
    print(image_emb.shape, gt.shape)
    test_dataset = TensorDataset(image_emb, gt)

    print("Train Dataset Length: ", len(train_dataset))
    print("Test Dataset Length: ", len(test_dataset))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_df, train_data_loader, test_data_loader


def mitigate_error_slices_waterbirds(args):
    args.input_shape = get_input_shape(args.dataset)
    clf = create_classifier(args, mode=args.mode)["classifier"]
    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    tr_land_df = pd.read_csv(args.clf_results_csv.format(args.seed, "valid", "landbirds"))
    va_land_df = pd.read_csv(args.clf_results_csv.format(args.seed, "test", "landbirds"))
    tr_water_df = pd.read_csv(args.clf_results_csv.format(args.seed, "valid", "waterbirds"))
    va_water_df = pd.read_csv(args.clf_results_csv.format(args.seed, "test", "waterbirds"))
    tr_df = pd.concat([tr_land_df, tr_water_df], axis=1)
    tr_df = tr_df.loc[:, ~tr_df.columns.duplicated()]
    va_df = pd.concat([va_land_df, va_water_df], axis=1)
    va_df = va_df.loc[:, ~va_df.columns.duplicated()]
    print(tr_df.shape, va_df.shape)
    final_csv_name = f"final_mitigation.csv"
    land_attrs = pickle.load(open(args.slice_names.format(args.seed, "landbirds"), "rb"))
    land_attrs = list(land_attrs.keys())
    print(land_attrs)
    water_attrs = pickle.load(open(args.slice_names.format(args.seed, "waterbirds"), "rb"))
    water_attrs = list(water_attrs.keys())
    print(water_attrs)
    attrs = land_attrs + water_attrs
    print(attrs)

    col_name_list = attrs
    col_name = []
    for col in col_name_list:
        col_name.append([col, col])

    for col in col_name:
        print(f"\n  ==================================== Hypothesis: {col} ====================================")
        test_df, train_data_loader, test_data_loader = generate_ds_last_layer_retrain(
            tr_df, va_df, args.clf_image_emb_path, args.batch_size, args.seed, n_samples=args.n,
            col_name_0=col[0], col_name_1=col[1])
        clf = create_classifier(args, mode=args.mode)["classifier"]
        optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        hyp_name = f"all_slices_y_ensemble_yes_land_hyp_{col[0]} | water_hyp_{col[1]}"

        model_name = args.save_path / f"{hyp_name}.pth"
        binary_predictions, _ = last_layer_retrain(
            clf, args.epochs, train_data_loader, test_data_loader, criterion, optimizer,
            args.device, model_name, batch_size=args.batch_size, loss_type="CE")
        test_df.loc[:, f"{hyp_name}_Predictions_bin"] = binary_predictions

        calculate_worst_group_acc_waterbirds(
            test_df, pred_col=f"{hyp_name}_Predictions_bin", attribute_col="attribute_bg_predict"
        )
        print(f"==================================== Hypothesis: {col} ==================================== \n")

    print("\n")
    cols = attrs

    water_birds_df = test_df[test_df["out_put_GT"] == 1]
    max_col = water_birds_df[water_attrs].idxmax(axis=1)
    pred_col = max_col.apply(
        lambda x: f"all_slices_y_ensemble_yes_land_hyp_{x} | water_hyp_{x}_Predictions_bin")
    water_birds_df['all_slices_y_ensemble_y_pred'] = water_birds_df.apply(lambda row: row[pred_col[row.name]],
                                                                          axis=1)

    land_birds_df = test_df[test_df["out_put_GT"] == 0]
    max_col = land_birds_df[land_attrs].idxmax(axis=1)
    pred_col = max_col.apply(
        lambda x: f"all_slices_y_ensemble_yes_land_hyp_{x} | water_hyp_{x}_Predictions_bin")
    land_birds_df['all_slices_y_ensemble_y_pred'] = land_birds_df.apply(lambda row: row[pred_col[row.name]],
                                                                        axis=1)
    test_df = pd.concat([water_birds_df, land_birds_df], axis=0)
    prediction_col = "all_slices_y_ensemble_y_pred"

    print("#################################### Ground truth slice ###########################################")
    calculate_worst_group_acc_waterbirds(
        test_df, pred_col=prediction_col, attribute_col="attribute_bg_predict", log_file=args.out_file
    )
    print("#################################### Ground truth slice ###########################################")

    print(args.save_path / final_csv_name)
    test_df.to_csv(args.save_path / final_csv_name, index=False)


def mitigate_error_slices_celebA(args):
    tr_df = pd.read_csv(args.clf_results_csv.format(args.seed, "valid"))
    va_df = pd.read_csv(args.clf_results_csv.format(args.seed, "test"))
    args.slice_names = Path(args.slice_names.format(args.seed))
    args.input_shape = get_input_shape(args.dataset)
    clf = create_classifier(args, mode=args.mode)["classifier"]
    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    final_csv_name = f"final_mitigation.csv"
    attrs = pickle.load(open(args.slice_names, "rb"))
    col_name = []
    col_name_list = list(attrs.keys())
    for col in col_name_list:
        col_name.append([col, col])
    for col in col_name:
        print(f"\n  ==================================== Hypothesis: {col} ====================================")
        test_df, train_data_loader, test_data_loader = generate_ds_last_layer_retrain(
            tr_df, va_df, args.clf_image_emb_path, args.batch_size, args.seed, n_samples=args.n,
            col_name_0=col[0], col_name_1=col[1])

        clf = create_classifier(args, mode=args.mode)["classifier"]
        optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        hyp_name = f"all_slices_y_ensemble_yes_{col[0]}"

        model_name = args.save_path / f"{hyp_name}.pth"
        binary_predictions, _ = last_layer_retrain(
            clf, args.epochs, train_data_loader, test_data_loader, criterion, optimizer,
            args.device, model_name, batch_size=args.batch_size, loss_type="CE")
        test_df.loc[:, f"{hyp_name}_Predictions_bin"] = binary_predictions

        print(test_df)
        print(f"==================================== Hypothesis: {col} ==================================== \n")

    print("\n")
    print("-------------------------------------------------------------------------------------------")
    print("#################################### Results ###########################################")
    print("-------------------------------------------------------------------------------------------")
    cols = list(attrs.keys())
    print("################################################################################################")
    print("############################### Ensemble predictions ########################################")
    print("################################################################################################")
    max_col = test_df[cols].idxmax(axis=1)
    pred_col = max_col.apply(lambda x: f"all_slices_y_ensemble_yes_{x}_Predictions_bin")
    test_df['all_slices_y_ensemble_y_pred'] = test_df.apply(lambda row: row[pred_col[row.name]], axis=1)
    pos_pred_col = "all_slices_y_ensemble_y_pred"
    neg_pred_col = "Predictions_bin"

    print("------------------------------------------------------------------------------------------------------")
    print("#################################### GT slices ###########################################")
    calculate_worst_group_acc_celebA(test_df, pos_pred_col, neg_pred_col, attribute_col="attribute_bg_predict",
                                     print_non_blonde=False, log_file=args.out_file)
    print("------------------------------------------------------------------------------------------------------")
    print("\n")
    print(args.save_path / final_csv_name)
    test_df.to_csv(args.save_path / final_csv_name, index=False)
    test_df.to_csv(args.save_path / final_csv_name, index=False)


def mitigate_error_slices_rsna(args):
    tr_df = pd.read_csv(args.clf_results_csv.format(args.seed, "valid"))
    va_df = pd.read_csv(args.clf_results_csv.format(args.seed, "test"))

    args.slice_names = Path(args.slice_names.format(args.seed))
    args.input_shape = get_input_shape(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    final_csv_name = f"final_mitigation.csv"
    attrs = pickle.load(open(args.slice_names, "rb"))

    col_name_list = list(attrs.keys())
    col_name = []
    for col in col_name_list:
        col_name.append([col, col])

    for col in col_name:
        print(f"\n  ==================================== Hypothesis: {col} ====================================")
        test_df, train_data_loader, test_data_loader = generate_ds_last_layer_retrain(
            tr_df, va_df, args.clf_image_emb_path, args.batch_size, args.seed, n_samples=args.n,
            col_name_0=col[0], col_name_1=col[1])

        clf = create_classifier(args, mode=args.mode).classifier
        optimizer = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        hyp_name = f"all_slices_y_ensemble_yes_{col[0]}"

        model_name = args.save_path / f"{hyp_name}.pth"
        binary_predictions, proba = last_layer_retrain(
            clf, args.epochs, train_data_loader, test_data_loader, criterion, optimizer,
            args.device, model_name, batch_size=args.batch_size, loss_type="BCE")
        test_df.loc[:, f"{hyp_name}_Predictions_bin"] = binary_predictions
        test_df.loc[:, f"{hyp_name}_Predictions_proba"] = proba

        print(test_df)
        acc_worst_group = calculate_worst_group_acc_rsna_mammo(
            test_df, pred_col=f"{hyp_name}_Predictions", attribute_col="calc")
        print(f"Avg. accuracy worst group: {acc_worst_group}")
        print(f"==================================== Hypothesis: {col} ==================================== \n")

    print("\n")
    print("-------------------------------------------------------------------------------------------")
    print("#################################### Results ###########################################")
    print("-------------------------------------------------------------------------------------------")
    cols = list(attrs.keys())
    print("################################################################################################")
    print("############################### Ensemble predictions ########################################")
    print("################################################################################################")
    max_col = test_df[cols].idxmax(axis=1)
    pred_col = max_col.apply(lambda x: f"all_slices_y_ensemble_yes_{x}_Predictions")
    test_df['all_slices_y_ensemble_y_pred_bin'] = test_df.apply(lambda row: row[f"{pred_col[row.name]}_bin"],
                                                                axis=1)
    test_df['all_slices_y_ensemble_y_pred_proba'] = test_df.apply(lambda row: row[f"{pred_col[row.name]}_proba"],
                                                                  axis=1)
    pos_pred_col = "all_slices_y_ensemble_y_pred_proba"
    neg_pred_col = "out_put_predict"

    print("------------------------------------------------------------------------------------------------------")
    print("############################### Ground truth slices ########################################")
    calculate_worst_group_acc_med_img(
        test_df, pos_pred_col=pos_pred_col, neg_pred_col=neg_pred_col, attribute_col="calc", log_file=args.out_file,
        disease="Cancer")
    print("------------------------------------------------------------------------------------------------------")

    print(test_df.columns)
    test_df.to_csv(args.save_path / final_csv_name, index=False)


def mitigate_error_slices_nih(args):
    tr_df = pd.read_csv(args.clf_results_csv.format(args.seed, "valid"))
    va_df = pd.read_csv(args.clf_results_csv.format(args.seed, "test"))

    args.slice_names = Path(args.slice_names.format(args.seed))
    args.input_shape = get_input_shape(args.dataset)
    clf = create_classifier(args, mode=args.mode)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss_type = "BCE"
    optimizer = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_csv_name = f"final_mitigation.csv"
    attrs = pickle.load(open(args.slice_names, "rb"))

    col_name_list = list(attrs.keys())
    col_name = []
    for col in col_name_list:
        col_name.append([col, col])
    for col in col_name:
        print(f"\n  ==================================== Hypothesis: {col} ====================================")
        test_df, train_data_loader, test_data_loader = generate_ds_last_layer_retrain(
            tr_df, va_df, args.clf_image_emb_path, args.batch_size, args.seed, n_samples=args.n,
            col_name_0=col[0], col_name_1=col[1])

        clf = create_classifier(args, mode=args.mode)
        optimizer = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        hyp_name = f"all_slices_y_ensemble_yes_{col[0]}"

        model_name = args.save_path / f"{hyp_name}.pth"
        binary_predictions, proba = last_layer_retrain(
            clf, args.epochs, train_data_loader, test_data_loader, criterion, optimizer,
            args.device, model_name, batch_size=args.batch_size, loss_type=loss_type)
        test_df.loc[:, f"{hyp_name}_Predictions_bin"] = binary_predictions
        test_df.loc[:, f"{hyp_name}_Predictions_proba"] = proba

        print(test_df)
        print(f"==================================== Hypothesis: {col} ==================================== \n")

    print("\n")
    print("-------------------------------------------------------------------------------------------")
    print("#################################### Results ###########################################")
    print("-------------------------------------------------------------------------------------------")
    cols = list(attrs.keys())

    print("################################################################################################")
    print("############################### Ensemble predictions ########################################")
    print("################################################################################################")
    max_col = test_df[cols].idxmax(axis=1)
    pred_col = max_col.apply(lambda x: f"all_slices_y_ensemble_yes_{x}_Predictions")
    test_df['all_slices_y_ensemble_y_pred_bin'] = test_df.apply(lambda row: row[f"{pred_col[row.name]}_bin"], axis=1)
    test_df['all_slices_y_ensemble_y_pred_proba'] = test_df.apply(lambda row: row[f"{pred_col[row.name]}_proba"],
                                                                  axis=1)
    pos_pred_col = "all_slices_y_ensemble_y_pred_proba"
    neg_pred_col = "out_put_predict"

    print("------------------------------------------------------------------------------------------------------")
    print("############################### Ground truth slices ########################################")
    acc_worst_group_tube = calculate_worst_group_acc_med_img(
        test_df, pos_pred_col=pos_pred_col, neg_pred_col=neg_pred_col, attribute_col="tube", log_file=args.out_file,
        disease="Pneumothorax")
    print("------------------------------------------------------------------------------------------------------")

    print("\n")
    print("------------------------------------------------------------------------------------------------------")

    test_df.to_csv(args.save_path / final_csv_name, index=False)
    print(f"file saved at: {args.save_path / final_csv_name}")


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Waterbirds", type=str)
    parser.add_argument("--n", default=200, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--classifier", default="ResNet50", type=str, help="n_samples")
    parser.add_argument("--slice_names", default="", type=str, help="pkl file containing the slice names")
    parser.add_argument(
        "--classifier_check_pt", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl",
        help=""
    )
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help=""
    )
    parser.add_argument(
        "--clf_results_csv", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_dataframe_mitigation.csv",
        help=""
    )
    parser.add_argument(
        "--clf_image_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy",
        help=""
    )
    parser.add_argument(
        "--mode", default="last_layer_retrain",
        help="last_layer_retrain with validation set (in Kirichenko et al., 2021) or last layer finetune with train set"
    )
    parser.add_argument('--default hypothesis', type=str, nargs='+',
                        help='A list of hypothesis, each optionally containing spaces and underscores')

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.save_path = Path(args.save_path.format(args.seed))
    args.save_path.mkdir(parents=True, exist_ok=True)
    args.out_file = args.save_path / f"ladder_mitigate_slices.txt"
    print("\n")
    print(args.save_path)
    if args.dataset.lower() == "waterbirds":
        mitigate_error_slices_waterbirds(args)
    elif args.dataset.lower() == "celeba":
        mitigate_error_slices_celebA(args)
    elif args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr":
        mitigate_error_slices_rsna(args)
    elif args.dataset.lower() == "nih":
        mitigate_error_slices_nih(args)

    print(f"log saved at: {args.out_file}")
    # Specify the path to your file
    file_path = args.out_file

    # Open the file in read mode and print its content
    print(f"####### Overall dataset performance: #######")
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)


if __name__ == "__main__":
    _args = config()
    main(_args)
