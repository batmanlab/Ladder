import gc
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from Aligners.single_layer_network import Single_layer_network
from Classifiers.models.EfficientNet import MammoModel_Efficient_net
from Datasets.dataset_utils import get_dataset
from Detectors.retinanet.detector_model import RetinaNet_efficientnet
from metrics import pr_auc, auroc, pfbeta_binarized, compute_accuracy, compute_auprc
from utils import seed_all, AverageMeter, timeSince, get_gt_pred

warnings.filterwarnings("ignore")
import numpy as np


def train(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    args.model_base_name = 'efficientv2_s' if 'efficientnetv2' in args.arch else 'efficientnetb5'
    print(f"df shape: {args.df.shape}")
    oof_df = pd.DataFrame()
    for fold in range(args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        args.train_folds = args.df[args.df['fold'] != args.cur_fold].reset_index(drop=True)
        args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)
        _oof_df = train_loop(args, device)
        oof_df = pd.concat([oof_df, _oof_df])
    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_csv(args.output_path / f'seed_{args.seed}_outputs.csv', index=False)
    print(oof_df.shape)
    # print(oof_df)

    print('================ CV ================')
    args.valid_folds = oof_df
    gt, pred = get_gt_pred(args)
    score = pfbeta_binarized(gt=gt, pred=pred)
    prauc = pr_auc(gt=gt, pred=pred)
    aucroc = auroc(gt=gt, pred=pred)
    auprc = compute_auprc(gt=gt, pred=pred)
    acc = compute_accuracy(gt=torch.from_numpy(gt), pred=torch.from_numpy((pred >= 0.5).astype(int)))

    print(f'Score: {score}, acc: {acc}%, PR-AUC: {prauc}, AUC-ROC: {aucroc}, AUPRC: {auprc}')
    print(f'Checkpoints saved at: {args.chk_pt_path}')
    print(f'Outputs saved at: {args.output_path}')


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        args.train_folds = args.train_folds.head(10)
        args.valid_folds = args.valid_folds.sample(n=300)
        # print(args.valid_folds[args.valid_folds["Architectural_Distortion"] == 1].shape)

    train_loader, valid_loader = get_dataset(args)
    print(f'train_loader: {len(train_loader)}', f'valid_loader: {len(valid_loader)}')

    concept_model = None
    if args.concept_model_type.lower() == 'classification':
        concept_model = MammoModel_Efficient_net(args.arch, pretrained=True, get_features=True)
        args.concept_classifier_model = f"{args.concept_classifier_model_name}_seed_{args.seed}_fold_{args.cur_fold}_best_auroc.pth"
        concept_model.load_state_dict(
            torch.load(args.concept_classifier_checkpoint_path / args.concept_classifier_model)['state_dict']
        )
    elif args.concept_model_type.lower() == 'detection':
        concept_model = RetinaNet_efficientnet(
            num_classes=1, model_type=args.arch, focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma, domain_classifier=True
        )
        args.concept_classifier_model = f"{args.concept_classifier_model_name}_seed_{args.seed}_fold_0_best_auroc.pth"
        concept_model.load_state_dict(
            torch.load(args.concept_classifier_checkpoint_path / args.concept_classifier_model)['state_dict']
        )

    concept_model.eval()
    concept_model.to(device)

    domain_classifier = Single_layer_network(input_size=args.concept_model_feature_size, output_size=1).to(device)
    optimizer = Adam(domain_classifier.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.epochs_warmup, num_training_steps=args.epochs, num_cycles=args.num_cycles
    )
    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    best_score = 0.
    best_aucroc = 0.
    best_prauc = 0.
    best_acc = 0.
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_train_loss = train_fn(
            args, train_loader, domain_classifier, concept_model, criterion, optimizer, epoch, device, logger
        )
        scheduler.step()
        avg_val_loss, predictions = valid_fn(
            valid_loader, domain_classifier, concept_model, criterion, epoch, args, device, logger
        )

        args.valid_folds['prediction'] = predictions
        gt, pred = get_gt_pred(args)
        score = pfbeta_binarized(gt=gt, pred=pred)
        prauc = pr_auc(gt=gt, pred=pred)
        aucroc = auroc(gt=gt, pred=pred)
        auprc = compute_auprc(gt=gt, pred=pred)

        acc = compute_accuracy(gt=torch.from_numpy(gt), pred=torch.from_numpy((pred >= 0.5).astype(int)))

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
            f'AUC-ROC Score: {aucroc:.4f}, AUPRC Score: {auprc:.4f}'
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
                    'state_dict': domain_classifier.state_dict(),
                    'predictions': predictions,
                    'auroc': aucroc,
                    'prauc': prauc,
                    'pF': score,
                    'acc': acc,
                    'auprc': auprc
                },
                args.chk_pt_path / f'{args.concept}_seed_{args.seed}_fold_{args.cur_fold}_best_auroc.pth'
            )

    predictions = torch.load(
        args.chk_pt_path / f'{args.concept}_seed_{args.seed}_fold_{args.cur_fold}_best_auroc.pth',
        map_location='cpu')['predictions']
    args.valid_folds['prediction'] = predictions
    print(
        f'[Fold{args.cur_fold}] Best pF: {best_score}, Best Acccuracy: {best_acc}%, Best PR-AUC: {best_prauc}, '
        f'Best AUC-ROC: {best_aucroc:.4f}'
    )

    torch.cuda.empty_cache()
    gc.collect()
    valid_folds = args.valid_folds.copy()
    return valid_folds


def train_fn(args, train_loader, domain_classifier, concept_model, criterion, optimizer, epoch, device, logger):
    domain_classifier.train()

    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = time.time()
    for step, data in enumerate(train_loader):
        x = data['x'].to(device)
        y = data['y'].float().to(device)
        batch_size = y.size(0)
        with torch.no_grad():
            features, _ = concept_model(x)
        with torch.cuda.amp.autocast(enabled=args.apex):
            y_hat = domain_classifier(features)
        loss = criterion(y_hat.view(-1, 1), y.view(-1, 1))
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # batch scheduler
        # scheduler.step()
        end = time.time()
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


def valid_fn(valid_loader, domain_classifier, concept_model, criterion, epoch, args, device, logger):
    losses = AverageMeter()
    domain_classifier.eval()
    preds = []
    start = time.time()
    for step, data in enumerate(valid_loader):
        x = data['x'].to(device)
        y = data['y'].float().to(device)
        batch_size = y.size(0)
        with torch.no_grad():
            features, _ = concept_model(x)
            y_hat = domain_classifier(features)
        loss = criterion(y_hat.view(-1, 1), y.view(-1, 1))
        losses.update(loss.item(), batch_size)
        preds.append(y_hat.squeeze(1).sigmoid().to('cpu').numpy())
        end = time.time()
        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

        if step % args.log_freq == 0 or step == (len(valid_loader) - 1):
            index = step + len(valid_loader) * epoch
            logger.add_scalar('valid/iter_loss', losses.avg, index)

    predictions = np.concatenate(preds)
    return losses.avg, predictions
