import gc
import os
import re
import time
import warnings
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn as nn
from pydicom import dcmread
# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from Classifiers.models.EfficientNet import MammoModel_Efficient_net
from Classifiers.models.Set_Transformer import SetTransformer
from Datasets.dataset_utils import get_dataset
from Plots.plots import plot_grad_cam_image_with_bounding_boxes_and_heatmap
from metrics import pr_auc, auroc, pfbeta_binarized, compute_accuracy
from utils import seed_all, AverageMeter, timeSince, get_gt_pred
from tqdm import tqdm

warnings.filterwarnings("ignore")
import numpy as np


def train_UPMC(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    args.model_base_name = 'efficientv2_s' if 'efficientnetv2' in args.arch else 'efficientnetb5'
    print(f"df shape: {args.df.shape}")

    preds = []
    labels = []
    for fold in range(args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        args.train_folds = args.df[args.df['fold'] != args.cur_fold].reset_index(drop=True)
        args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)

        predictions, ground_truth = train_loop(args, device)

        np.save(args.output_path / f'predictions_fold_{str(args.cur_fold)}', predictions)
        np.save(args.output_path / f'ground_truth_fold_{str(args.cur_fold)}', ground_truth)
        preds.append(predictions)
        labels.append(ground_truth)

    pred = np.concatenate(preds)
    gt = np.concatenate(labels)
    np.save(args.output_path / 'predictions_full', pred)
    np.save(args.output_path / 'ground_truth_full', gt)

    print('================ CV ================')
    print(pred.shape, gt.shape)
    score = pfbeta_binarized(gt=gt, pred=pred)
    prauc = pr_auc(gt=gt, pred=pred)
    aucroc = auroc(gt=gt, pred=pred)
    acc = compute_accuracy(gt=torch.from_numpy(gt), pred=torch.from_numpy((pred >= 0.5).astype(int)))

    print(f'Score: {score}, acc: {acc}%, PR-AUC: {prauc}, AUC-ROC: {aucroc}')
    print(f'Checkpoints saved at: {args.chk_pt_path}')
    print(f'Outputs saved at: {args.output_path}')


def train_Vindr(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    args.model_base_name = 'efficientv2_s' if 'efficientnetv2' in args.arch else 'efficientnetb5'
    print(f"df shape: {args.df.shape}")
    oof_df = pd.DataFrame()
    # args.cur_fold = 0
    # args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
    # args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)
    #
    for fold in range(args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        args.train_folds = args.df[args.df['fold'] != args.cur_fold].reset_index(drop=True)
        args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)

        if args.inference_mode == "n":
            train_loop(args, device)
        elif args.inference_mode == "y":
            inference_loop(args, device)
    print(f'Checkpoints saved at: {args.chk_pt_path}')
    print(f'Outputs saved at: {args.output_path}')


def test_upmc(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    args.model_base_name = 'efficientv2_s' if 'efficientnetv2' in args.arch else 'efficientnetb5'
    print(f"df shape: {args.df.shape}")

    preds = []
    labels = []
    study_ids = []
    lats = []
    for fold in range(args.n_folds):
        args.cur_fold = fold
        print(f"================================= fold: {args.cur_fold} testing =================================")
        seed_all(args.seed)
        args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)

        if args.running_interactive and args.model_type.lower() == "concept-classifier-set":
            study_id_temp = [40421851, 936277906, 701722701, 936993559, 40357714, 936380516, 936168100, 936976095]
            args.valid_folds = args.valid_folds[args.valid_folds['STUDY_ID'].isin(study_id_temp)]
        elif args.running_interactive and args.model_type.lower() == "concept-classifier":
            args.valid_folds = args.valid_folds.sample(n=300)

        valid_loader = get_dataset(args, train=False)
        print(args.valid_folds.shape)
        print(f'valid_loader: {len(valid_loader)}')

        model, criterion = init(args, device, testing=True)
        _, np_preds, np_labels, np_study_id, np_lats = valid_fn(
            valid_loader, model, criterion, epoch=0, args=args, device=device
        )

        preds.append(np_preds)
        labels.append(np_labels)
        study_ids.append(np_study_id)
        lats.append(np_lats)

        print(np_preds.shape, np_labels.shape, np_study_id.shape, np_lats.shape)
        np.save(args.output_path / f'UPMC_fold_{args.cur_fold}_study_id.npy', np_study_id)
        np.save(args.output_path / f'UPMC_fold_{args.cur_fold}_lat.npy', np_lats)

    np_preds = np.concatenate(preds)
    np_labels = np.concatenate(labels)
    np_study_id = np.concatenate(study_ids)
    np_lats = np.concatenate(lats)

    df = pd.DataFrame({
        'STUDY_ID': np_study_id,
        'laterality': np_lats,
        f'{args.concept}_gt': np_labels,
        f'{args.concept}_prediction': np_preds,
    })

    df.to_csv(args.output_path / f'UPMC_{args.concept}.csv', index=False)
    print(df.shape)
    print(f'\nOutputs saved at: {args.output_path}')


def init(args, device, testing=True):
    model = None
    if args.model_type.lower() == 'concept-classifier':
        model = MammoModel_Efficient_net(args.arch, pretrained=True, get_features=False)
    elif args.model_type.lower() == 'concept-classifier-set':
        if args.arch.lower() == "tf_efficientnetv2_s":
            dim_input = 1280
        model = SetTransformer(args.arch, args.apex, dim_input=dim_input)

    model.to(device)
    if testing:
        model_checkpoint = torch.load(
            args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_acc_concept.pth'
        )
        model.load_state_dict(model_checkpoint['state_dict'])
        upmc_gt = model_checkpoint['gt']
        upmc_pred = model_checkpoint['predictions']
        score, prauc, aucroc, acc, acc_concept = calc_metrics(gt=upmc_gt, pred=upmc_pred)
        print(
            f'UPMC - pF Score: {score:.4f},  Accuracy: {acc:.4f}%, PR-AUC Score: {prauc:.4f}, '
            f'AUC-ROC Score: {aucroc:.4f}, Acc +ve Concept: {acc_concept:.4f}'
        )

    model.eval()
    print("Model loaded")
    wt = torch.tensor([args.class_weights[args.concept][f'fold_{str(args.cur_fold)}']]).to('cuda')
    criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=wt)

    return model, criterion


def calc_metrics(gt, pred):
    score = pfbeta_binarized(gt=gt, pred=pred)
    prauc = pr_auc(gt=gt, pred=pred)
    aucroc = auroc(gt=gt, pred=pred)
    acc = compute_accuracy(gt=torch.from_numpy(gt), pred=torch.from_numpy((pred >= 0.5).astype(int)))

    indices = np.where(gt == 1)[0]
    gt_1 = gt[indices]
    pred_1 = pred[indices]
    acc_concept = compute_accuracy(gt=torch.from_numpy(gt_1), pred=torch.from_numpy((pred_1 >= 0.5).astype(int)))
    return score, prauc, aucroc, acc, acc_concept


def test_ext(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    print(args.data_dir / args.csv_file)

    if args.target_dataset.lower() == "rsna":
        # args.df = args.df[args.df["cancer"] == 1]
        args.df.rename(columns={'patient_id': 'STUDY_ID'}, inplace=True)
        args.df.rename(columns={'view': 'view_position'}, inplace=True)
        args.df.rename(columns={'image_id': 'IMAGE_ID'}, inplace=True)
        args.df['IMAGE_ID'] = args.df['IMAGE_ID'].astype(str) + '.png'
    elif args.target_dataset.lower() == "vindr":
        # args.df = args.df[args.df["Skin_Retraction"] == 1]
        args.df.rename(columns={'study_id': 'STUDY_ID'}, inplace=True)
        args.df.rename(columns={'image_id': 'IMAGE_ID'}, inplace=True)
        args.df['IMAGE_ID'] = args.df['IMAGE_ID'].astype(str) + '.png'

    args.df = args.df.fillna(0)

    print(f"df columns: {list(args.df.columns)}")
    print(f"df shape: {args.df.shape}")

    seed_all(args.seed)
    args.model_base_name = 'efficientv2_s' if 'efficientnetv2' in args.arch else 'efficientnetb5'
    args.valid_folds = args.df
    print(f'valid: {args.valid_folds.shape}')
    print(args.valid_folds.columns)
    if args.running_interactive:
        args.valid_folds = args.valid_folds.head(20)

    predictions = []
    for fold in range(args.n_folds):
        args.cur_fold = fold
        print(f"================================= fold: {args.cur_fold} testing =================================")
        valid_loader = get_dataset(args, train=False)
        print(args.valid_folds.shape)
        print(f'valid_loader: {len(valid_loader)}')
        model, criterion = init(args, device, testing=True)
        _, np_preds, _, np_study_id, np_lats = valid_fn(
            valid_loader, model, criterion, epoch=0, args=args, device=device
        )

        np.save(args.output_path / f'fold_{args.cur_fold}_preds.npy', np_preds)
        np.save(args.output_path / f'fold_{args.cur_fold}_patient_id.npy', np_study_id)
        np.save(args.output_path / f'fold_{args.cur_fold}_lat.npy', np_lats)

        predictions.append(np_preds)

    patient_id = np.load(args.output_path / f'fold_{args.cur_fold}_patient_id.npy')
    lats = np.load(args.output_path / f'fold_{args.cur_fold}_lat.npy')

    np_predictions = np.stack(predictions, axis=0)
    mean_prediction = np.mean(np_predictions, axis=0)

    df = pd.DataFrame({
        'patient_id': patient_id,
        'laterality': lats,
        f'{args.concept}_fold_0': predictions[0],
        f'{args.concept}_fold_1': predictions[1],
        f'{args.concept}_fold_2': predictions[2],
        f'{args.concept}_fold_3': predictions[3],
        args.concept: mean_prediction
    })

    df.to_csv(args.output_path / f'{args.target_dataset}_{args.concept}.csv', index=False)
    print(f'\nOutputs saved at: {args.output_path}')


def create_output_csv(args, device):
    print("================== Creating output csv ======================")
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    oof_df = pd.DataFrame()
    for fold in range(args.n_folds):
        args.cur_fold = fold
        epoch = 1
        seed_all(args.seed)
        print(f'================== fold: {args.cur_fold} validating ======================')
        args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)
        if args.running_interactive:
            args.valid_folds = args.valid_folds.sample(n=1000)
        valid_loader = get_dataset(args, train=False)

        model = MammoModel_Efficient_net(args.arch, pretrained=True)
        model.to(device)
        model.load_state_dict(
            torch.load(
                args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_auroc.pth',
                map_location=device
            )['state_dict'])
        model.eval()
        wt = torch.tensor([args.wt_bce]).to('cuda')
        criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=wt)
        logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')

        avg_val_loss, predictions, _, _, _ = valid_fn(valid_loader, model, criterion, epoch, args, device, logger)

        args.valid_folds['prediction'] = predictions
        gt, pred = get_gt_pred(args)
        score = pfbeta_binarized(gt=gt, pred=pred)
        prauc = pr_auc(gt=gt, pred=pred)
        aucroc = auroc(gt=gt, pred=pred)
        acc = compute_accuracy(gt=torch.from_numpy(gt), pred=torch.from_numpy((pred >= 0.5).astype(int)))

        print(
            f'Epoch {epoch + 1} - pF Score: {score:.4f},  Accuracy: {acc:.4f}%, PR-AUC Score: {prauc:.4f}, '
            f'AUC-ROC Score: {aucroc:.4f}'
        )

        _oof_df = args.valid_folds.copy()
        oof_df = pd.concat([oof_df, _oof_df])
    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_csv(args.output_path / f'seed_{args.seed}_outputs.csv', index=False)
    print(oof_df.shape)
    print('================== Finishing output csv ======================')


def inference_loop(args, device):
    print('\n================== Saving Images ======================')
    print(f'================== fold: {args.cur_fold} saving ======================')
    op_path = args.output_path / args.target_dataset / args.concept
    os.makedirs(op_path, exist_ok=True)

    model = MammoModel_Efficient_net(args.arch, pretrained=True)
    model.load_state_dict(
        torch.load(args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_auroc.pth')[
            'state_dict'
        ])
    model.eval()

    target_layers = [model.model.blocks[-1][-1]]
    targets = [ClassifierOutputTarget(0)]
    # print(target_layers)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    args.batch_size = 1
    loader = get_dataset(args, train=False)

    print(f'Number of images: {len(loader)}')

    with tqdm(total=len(loader)) as t:
        for step, data in enumerate(loader):
            x = data['x'].to(device)
            y = data['y'].item()
            img_path = data['img_path'][0]

            grayscale_cams = cam(input_tensor=x, targets=targets, aug_smooth=False, eigen_smooth=True)
            threshold = 0.5  # Adjust the threshold as needed
            grayscale_cams_tensor = torch.tensor(grayscale_cams[0])
            binary_heatmap = torch.where(
                grayscale_cams_tensor > threshold,
                torch.ones_like(grayscale_cams_tensor), torch.zeros_like(grayscale_cams_tensor)
            )
            binary_heatmap_np = binary_heatmap.squeeze().cpu().numpy()
            contours, _ = cv2.findContours((binary_heatmap_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            best_x, best_y, best_w, best_h = 0, 0, 0, 0
            max_area = 0
            for contour in contours:
                xmin, ymin, w, h = cv2.boundingRect(contour)
                area = w * h
                # if area > max_area:
                best_x, best_y, best_w, best_h = xmin, ymin, w, h
                max_area = area

            split_img_name = img_path.split('/')
            study_id = split_img_name[-2]

            if args.dataset.lower() == 'upmc' and args.target_dataset.lower() == 'upmc':
                pattern = r'([\d.]+)\.png'
                match = re.search(pattern, split_img_name[-1])
                image_id = match.group(1)
            elif args.dataset.lower() == 'vindr' and args.target_dataset.lower() == 'vindr':
                image_id = split_img_name[-1].split('.')[0]

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            # image_name_w_bb = op_path / f'{study_id}_{image_id}_pred_w_bounding_box.png'
            # plot_grad_cam_image_with_bounding_boxes(img, best_x, best_y, best_w, best_h, image_name_w_bb, dpi=500)
            #
            # image_name_cam = op_path / f'{study_id}_{image_id}_pred_cam.png'
            # plot_grad_cam_image(img, heatmap=grayscale_cams[0], image_name=image_name_cam, dpi=500)

            if args.dataset.lower() == 'vindr' and args.target_dataset.lower() == 'vindr':
                bb_pred = [best_x, best_y, best_x + best_w, best_y + best_h]
                save_images_vindr(
                    args, y, study_id, image_id, png_img=img, heatmap=grayscale_cams[0], bounding_box_pred=bb_pred,
                    op_path=op_path
                )
            t.set_postfix(batch_id='{0}'.format(step + 1))
            t.update()

    torch.cuda.empty_cache()
    gc.collect()


def save_images_vindr(args, y, study_id, image_id, png_img, heatmap, bounding_box_pred, op_path):
    vindr_findings_path_parts = args.img_dir.split('/')
    vindr_findings_path = '/'.join(vindr_findings_path_parts[:-1])
    df = pd.read_csv(args.data_dir / vindr_findings_path / 'vindr_detection.csv')
    df = df[(df['study_id'] == study_id) & (df['image_id'] == image_id)]
    for index, row in df.iterrows():
        study_id = row['study_id']
        image_id = row['image_id']

        dicom_img = dcmread(f"{args.data_dir}/{vindr_findings_path}/images/{study_id}/{image_id}.dicom").pixel_array
        if y == 0:
            bounding_box_orig = [0, 0, 0, 0]
            bounding_box_gt = [0, 0, 0, 0]
        else:
            bounding_box_orig = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            bounding_box_gt = [row['resized_xmin'], row['resized_ymin'], row['resized_xmax'], row['resized_ymax']]
        image_name_w_bb = op_path / f'{args.concept}_{y}_{study_id}_{image_id}_{index}.png'
        plot_grad_cam_image_with_bounding_boxes_and_heatmap(
            dicom_img, png_img, heatmap, bounding_box_gt, bounding_box_orig, bounding_box_pred, image_name_w_bb,
            dpi=500, loc='upper right'
        )


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    if args.running_interactive and args.model_type.lower() == "concept-classifier-set":
        # test on small subsets of data on interactive mode
        study_ids = [40421851, 936277906, 701722701, 936993559, 40357714, 936380516, 936168100, 936976095]
        args.train_folds = args.train_folds[args.train_folds['STUDY_ID'].isin(study_ids)]
        args.valid_folds = args.valid_folds[args.valid_folds['STUDY_ID'].isin(study_ids)]
    elif args.running_interactive and args.model_type.lower() == "concept-classifier":
        # test on small subsets of data on interactive mode
        args.train_folds = args.train_folds.head(10)
        args.valid_folds = args.valid_folds.sample(n=300)

    train_loader, valid_loader = get_dataset(args)
    print(args.train_folds.shape, args.valid_folds.shape)
    print(f'train_loader: {len(train_loader)}', f'valid_loader: {len(valid_loader)}')

    model, criterion = init(args, device, testing=False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.epochs_warmup, num_training_steps=args.epochs, num_cycles=args.num_cycles
    )

    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')
    best_score = 0.
    best_aucroc = 0.
    best_prauc = 0.
    best_acc = 0.
    best_acc_concept = 0.

    for epoch in range(args.epochs):
        start_time = time.time()

        avg_train_loss = train_fn(args, train_loader, model, criterion, optimizer, epoch, device, logger)
        scheduler.step()
        avg_val_loss, pred, gt, _, _ = valid_fn(valid_loader, model, criterion, epoch, args, device, logger)
        score, prauc, aucroc, acc, acc_concept = calc_metrics(gt, pred)

        logger.add_scalar('train/epoch_loss', avg_train_loss, epoch)
        logger.add_scalar('valid/epoch_loss', avg_val_loss, epoch)
        logger.add_scalar('valid/epoch_aucroc', aucroc, epoch)
        logger.add_scalar('valid/epoch_prauc', prauc, epoch)
        logger.add_scalar('valid/epoch_pF', score, epoch)
        logger.add_scalar('valid/epoch_accuracy', acc, epoch)

        elapsed = time.time() - start_time
        print(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_train_loss:.4f},  avg_val_loss: {avg_val_loss:.4f},  '
            f'time: {elapsed:.0f}s'
        )
        print(
            f'Epoch {epoch + 1} - pF Score: {score:.4f},  Accuracy: {acc:.4f}%, PR-AUC Score: {prauc:.4f}, '
            f'AUC-ROC Score: {aucroc:.4f}, Acc +ve Concept: {acc_concept:.4f}'
        )

        if best_acc_concept < acc_concept:
            best_acc_concept = acc_concept
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_acc_concept.pth'
            print(f'Epoch {epoch + 1} - Save Best acc +ve concept: {best_acc_concept:.4f} Model')
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'predictions': pred,
                    'gt': gt,
                    'auroc': aucroc,
                    'prauc': prauc,
                    'pF': score,
                    'acc': acc,
                },
                args.chk_pt_path / model_name
            )

        if best_prauc < prauc:
            best_prauc = prauc
        if best_score < score:
            best_score = score
        if best_acc < acc:
            best_acc = acc
        if best_aucroc < aucroc:
            best_aucroc = aucroc
            print(f'Epoch {epoch + 1} - Save Best aucroc: {best_aucroc:.4f} Model')
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'predictions': pred,
                    'gt': gt,
                    'auroc': aucroc,
                    'prauc': prauc,
                    'pF': score,
                    'acc': acc,
                },
                args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_'
                                   f'fold_{args.cur_fold}_best_auroc.pth'
            )

    if os.path.exists(
            args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_acc_concept.pth'):
        model_chk_pt = torch.load(
            args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_acc_concept.pth',
            map_location='cpu')
    else:
        model_chk_pt = torch.load(
            args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_best_auroc.pth',
            map_location='cpu')

    predictions = model_chk_pt['predictions']
    ground_truth = model_chk_pt['gt']

    print(
        f'[Fold{args.cur_fold}] Best pF: {best_score}, Best Accuracy: {best_acc}%, Best PR-AUC: {best_prauc}, '
        f'Best AUC-ROC: {best_aucroc:.4f}, Acc +ve Concept: {best_acc_concept:.4f}'
    )

    torch.cuda.empty_cache()
    gc.collect()

    return predictions, ground_truth


def train_fn(args, train_loader, model, criterion, optimizer, epoch, device, logger):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = time.time()

    for step, data in enumerate(train_loader):
        x = data['x'].to(device)
        y = data['y'].float().to(device)
        batch_size = y.size(0)

        with torch.cuda.amp.autocast(enabled=args.apex):
            y_hat = model(x)

        loss = criterion(y_hat.view(-1, 1), y.view(-1, 1))
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar('train/epoch', epoch, index)
            logger.add_scalar('train/iter_loss', losses.avg, index)
            logger.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], index)

    return losses.avg


def valid_fn(valid_loader, model, criterion, epoch, args, device, logger=None):
    losses = AverageMeter()
    model.eval()
    study_ids = []
    lats = []
    preds = []
    labels = []
    start = time.time()

    with tqdm(total=len(valid_loader)) as t:
        for step, data in enumerate(valid_loader):
            x = data['x'].to(device)
            y = data['y'].float().to(device)
            study_id = data['study_id']
            laterality = data['laterality']

            batch_size = y.size(0)
            with torch.no_grad():
                y_hat = model(x)

            loss = criterion(y_hat.view(-1, 1), y.view(-1, 1))
            losses.update(loss.item(), batch_size)

            preds.append(y_hat.squeeze(1).sigmoid().to('cpu').numpy())
            labels.append(y.to('cpu').numpy())
            study_ids.append(np.array(study_id))
            lats.append(np.array(laterality))

            if (step % args.print_freq == 0 or step == (
                    len(valid_loader) - 1)
            ) and args.target_dataset.lower() != 'rsna':
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      .format(step, len(valid_loader),
                              loss=losses,
                              remain=timeSince(start, float(step + 1) / len(valid_loader))))

            if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and args.target_dataset.lower() != 'rsna':
                index = step + len(valid_loader) * epoch
                if logger is not None:
                    logger.add_scalar('valid/iter_loss', losses.avg, index)

            t.set_postfix(batch_id='{0}'.format(step + 1))
            t.update()

    np_preds = np.concatenate(preds)
    np_labels = np.concatenate(labels)
    np_study_id = np.concatenate(study_ids)
    np_lats = np.concatenate(lats)
    return losses.avg, np_preds, np_labels, np_study_id, np_lats
