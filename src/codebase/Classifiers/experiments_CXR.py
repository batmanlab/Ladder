import os
import pickle
import warnings
from os.path import join

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tqdm_base
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


# from tqdm.auto import tqdm


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def adjust_learning_rate(cfg, optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = cfg.lr
    if epoch in [10, 15, 20, 30]:
        print("Old lr: ", lr)
        lr /= 10
        print("New lr: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def uniform_binning(y_conf, bin_size=0.10):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    lower_bounds = upper_bounds - bin_size
    y_bin = []
    n_bins = len(upper_bounds)
    for s in y_conf:
        if (s <= upper_bounds[0]):
            y_bin.append(0)
        elif (s > lower_bounds[n_bins - 1]):
            y_bin.append(n_bins - 1)
        else:
            for i in range(1, n_bins - 1):
                if (s > lower_bounds[i]) & (s <= upper_bounds[i]):
                    y_bin.append(i)
                    break
    y_bin = np.asarray(y_bin)
    return y_bin

def test_mimic(args, test_loader):
    device = "cuda" if args.device == "cuda" else 'cpu'
    args.model.to(device)
    args.model.eval()
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    out_put_tubes = torch.FloatTensor().cuda()
    out_put_tube_prob = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, samples in enumerate(test_loader):
                images = samples["img"].to(device)
                targets = samples["lab"].to(device)
                names = np.asarray(samples["file_name"])
                tubes = samples["tube_label"].to(device)
                tube_prob = samples["tube_prob"].float().to(device)

                outputs = args.model(images)

                out_put_predict = torch.cat((out_put_predict, outputs), dim=0)
                out_put_GT = torch.cat((out_put_GT, targets), dim=0)
                out_put_tubes = torch.cat((out_put_tubes, tubes), dim=0)
                out_put_tube_prob = torch.cat((out_put_tube_prob, tube_prob), dim=0)

                if batch_idx == 0:
                    all_names = names
                else:
                    all_names = np.append(all_names, names, axis=0)

                t.set_postfix(batch_id='{0}'.format(batch_idx + 1))
                t.update()

    print("All data: ")
    out_AUROC_all, out_AUPRC_all = compute_AUC(gt=out_put_GT, pred=out_put_predict)
    print(f"AuROC: {out_AUROC_all}, AuRPC: {out_AUPRC_all}")
    print(" ")

    # Get patients without pneumothorax
    neg_pneumothorax_idx = torch.nonzero(out_put_GT.cpu() == 0, as_tuple=True)[0]
    out_put_GT_neg_pt = out_put_GT[neg_pneumothorax_idx]
    out_put_predict_neg_pt = out_put_predict[neg_pneumothorax_idx]

    print("Patients with (+ve tubes and +ve pneumothorax) and all -ve patients")
    # Get patients with tubes
    tube_indices = torch.nonzero(out_put_tubes.cpu() == 1, as_tuple=True)[0]
    out_put_predict_tubes = out_put_predict[tube_indices]
    out_put_GT_tubes = out_put_GT[tube_indices]

    # Get patients with tubes and with pneumothorax
    pos_pneumothorax_idx = torch.nonzero(out_put_GT_tubes.cpu() == 1, as_tuple=True)[0]
    out_put_GT_tubes_pos_pt = out_put_GT_tubes[pos_pneumothorax_idx]
    out_put_predict_tubes_pos_pt = out_put_predict_tubes[pos_pneumothorax_idx]
    print(out_put_GT_tubes_pos_pt.size(), out_put_predict_tubes_pos_pt.size())

    out_put_GT_tubes_ds = torch.cat((out_put_GT_tubes_pos_pt, out_put_GT_neg_pt), dim=0)
    out_put_predict_tubes_ds = torch.cat((out_put_predict_tubes_pos_pt, out_put_predict_neg_pt), dim=0)

    print(out_put_GT_tubes_ds.size(), out_put_predict_tubes_ds.size())
    auroc_pos_tube, aurpc_pos_tube = compute_AUC(gt=out_put_GT_tubes_ds, pred=out_put_predict_tubes_ds)
    print(f"AuROC: {auroc_pos_tube}, AuRPC: {aurpc_pos_tube}")
    print(" ")

    print("Patients with (-ve tubes and +ve pneumothorax) and all -ve patients")
    # Get patients without tubes
    no_tube_indices = torch.nonzero(out_put_tubes.cpu() == 0, as_tuple=True)[0]
    out_put_predict_no_tubes = out_put_predict[no_tube_indices]
    out_put_GT_no_tubes = out_put_GT[no_tube_indices]

    # Get patients without tubes and with pneumothorax
    pos_pt_ix = torch.nonzero(out_put_GT_no_tubes.cpu() == 1, as_tuple=True)[0]
    out_put_GT_no_tubes_pos_pt = out_put_GT_no_tubes[pos_pt_ix]
    out_put_predict_no_tubes_pos_pt = out_put_predict_no_tubes[pos_pt_ix]
    print(out_put_GT_no_tubes_pos_pt.size(), out_put_predict_no_tubes_pos_pt.size())

    out_put_GT_no_tubes_ds = torch.cat((out_put_GT_no_tubes_pos_pt, out_put_GT_neg_pt), dim=0)
    out_put_predict_no_tubes_ds = torch.cat((out_put_predict_no_tubes_pos_pt, out_put_predict_neg_pt), dim=0)
    print(out_put_GT_no_tubes_ds.size(), out_put_predict_no_tubes_ds.size())
    auroc_no_tubes, aurpc_no_tubes = compute_AUC(gt=out_put_GT_no_tubes_ds, pred=out_put_predict_no_tubes_ds)

    print(f"AuROC: {auroc_no_tubes}, AuRPC: {aurpc_no_tubes}")
    print(" ")

    torch.save(out_put_GT.cpu(), args.output_path / "GT.pth.tar")
    torch.save(out_put_predict.cpu(), args.output_path / "predictions.pth.tar")
    torch.save(out_put_tubes.cpu(), args.output_path / "tubes.pth.tar")
    torch.save(out_put_tube_prob.cpu(), args.output_path / "tube_prob.pth.tar")
    np.save(args.output_path / "image_names.npy", all_names)

    print(f"Outputs are saved at:{args.output_path}")

    df = pd.DataFrame({
        "Image_name": all_names,
        "GT": out_put_GT.squeeze().cpu().numpy().astype(int),
        "Predictions": torch.nn.Sigmoid()(out_put_predict.squeeze()).cpu().numpy(),
        "Tubes": out_put_tubes.squeeze().cpu().numpy().astype(int),
        "Predictions_bin": (torch.nn.Sigmoid()(out_put_predict.squeeze()).cpu().numpy() >= 0.5).astype(int)
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
    print(
        f"[Acc] overall pneumothorax: {overall_pneumothorax_acc}, "
        f"[Acc] pneumothorax with tube: {pneumothorax_w_tube_acc}, "
        f"[Acc] pneumothorax without tube: : {pneumothorax_wo_tube_acc}"
    )

    csv_file_path = args.output_path / "test_additional_info.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Saved additional info to {csv_file_path}")

def test(args, test_loader):
    # # pred = torch.load("/ocean/projects/asc170022p/shg121/PhD/Multimodal-mistakes-debug/out/NIH/zz/Classifier/ResNet50/Pneumothorax/img_size_224_lr_1e-05_epochs_60_loss_BCE_W/predictions.pth.tar")
    # # print(torch.nn.Sigmoid()(pred.squeeze()))
    # # pred_disease = (torch.sigmoid(pred) > 0.5).int().squeeze()
    # # print(pred_disease[pred_disease == 1])
    # proba = torch.nn.Sigmoid()(pred.squeeze())
    # pred = (proba >= 0.5).int()
    # print(pred[pred == 1])
    df = pd.read_csv(args.output_path / f"test_additional_info.csv")
    print(df.head())
    AUROCs = roc_auc_score(df["GT"].values, df["Predictions"].values)
    print(f"Overall AuROC: {AUROCs}")
    pneumothorax_df = df[df["GT"] == 1]
    print(pneumothorax_df.shape)
    overall_pneumothorax_acc = pneumothorax_df[pneumothorax_df["Predictions_bin"] == 1].shape[0] / \
                               pneumothorax_df.shape[0]
    pneumothorax_w_tube_df = df[(df["GT"] == 1) & ((df["Tubes"] == 1))]
    pneumothorax_w_tube_acc = pneumothorax_w_tube_df[pneumothorax_w_tube_df["Predictions_bin"] == 1].shape[0] / \
                              pneumothorax_w_tube_df.shape[0]
    pneumothorax_wo_tube_df = df[(df["GT"] == 1) & ((df["Tubes"] == 0))]
    pneumothorax_wo_tube_acc = pneumothorax_wo_tube_df[pneumothorax_wo_tube_df["Predictions_bin"] == 1].shape[0] / \
                               pneumothorax_wo_tube_df.shape[0]

    corr = df[(df["GT"] == 1) & (df["Predictions_bin"] == 1)]
    incorr = df[(df["GT"] == 1) & (df["Predictions_bin"] == 0)]
    print(f"[Shape] corr: {corr.shape[0]}, incorr: {incorr.shape[0]}")
    print(f"[Shape] tube: {pneumothorax_w_tube_df.shape[0]}, no tube: {pneumothorax_wo_tube_df.shape[0]}")
    print(
        f"[Acc] overall pneumothorax: {overall_pneumothorax_acc}, "
        f"[Acc] pneumothorax with tube: {pneumothorax_w_tube_acc}, "
        f"[Acc] pneumothorax without tube: : {pneumothorax_wo_tube_acc}"
    )

    device = "cuda" if args.device == "cuda" else 'cpu'
    args.model.to(device)
    args.model.eval()
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    out_put_tubes = torch.FloatTensor().cuda()
    out_put_tube_prob = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, samples in enumerate(test_loader):
                images = samples["img"].to(device)
                targets = samples["lab"].to(device)
                names = np.asarray(samples["file_name"])
                tubes = samples["tube_label"].to(device)
                tube_prob = samples["tube_prob"].float().to(device)

                outputs = args.model(images)

                out_put_predict = torch.cat((out_put_predict, outputs), dim=0)
                out_put_GT = torch.cat((out_put_GT, targets), dim=0)
                out_put_tubes = torch.cat((out_put_tubes, tubes), dim=0)
                out_put_tube_prob = torch.cat((out_put_tube_prob, tube_prob), dim=0)

                if batch_idx == 0:
                    all_names = names
                else:
                    all_names = np.append(all_names, names, axis=0)

                t.set_postfix(batch_id='{0}'.format(batch_idx + 1))
                t.update()

    print("All data: ")
    out_AUROC_all, out_AUPRC_all = compute_AUC(gt=out_put_GT, pred=out_put_predict)
    print(f"AuROC: {out_AUROC_all}, AuRPC: {out_AUPRC_all}")
    print(" ")

    # Get patients without pneumothorax
    neg_pneumothorax_idx = torch.nonzero(out_put_GT.cpu() == 0, as_tuple=True)[0]
    out_put_GT_neg_pt = out_put_GT[neg_pneumothorax_idx]
    out_put_predict_neg_pt = out_put_predict[neg_pneumothorax_idx]

    print("Patients with (+ve tubes and +ve pneumothorax) and all -ve patients")
    # Get patients with tubes
    tube_indices = torch.nonzero(out_put_tubes.cpu() == 1, as_tuple=True)[0]
    out_put_predict_tubes = out_put_predict[tube_indices]
    out_put_GT_tubes = out_put_GT[tube_indices]

    # Get patients with tubes and with pneumothorax
    pos_pneumothorax_idx = torch.nonzero(out_put_GT_tubes.cpu() == 1, as_tuple=True)[0]
    out_put_GT_tubes_pos_pt = out_put_GT_tubes[pos_pneumothorax_idx]
    out_put_predict_tubes_pos_pt = out_put_predict_tubes[pos_pneumothorax_idx]
    print(out_put_GT_tubes_pos_pt.size(), out_put_predict_tubes_pos_pt.size())

    out_put_GT_tubes_ds = torch.cat((out_put_GT_tubes_pos_pt, out_put_GT_neg_pt), dim=0)
    out_put_predict_tubes_ds = torch.cat((out_put_predict_tubes_pos_pt, out_put_predict_neg_pt), dim=0)

    print(out_put_GT_tubes_ds.size(), out_put_predict_tubes_ds.size())
    auroc_pos_tube, aurpc_pos_tube = compute_AUC(gt=out_put_GT_tubes_ds, pred=out_put_predict_tubes_ds)
    print(f"AuROC: {auroc_pos_tube}, AuRPC: {aurpc_pos_tube}")
    print(" ")

    print("Patients with (-ve tubes and +ve pneumothorax) and all -ve patients")
    # Get patients without tubes
    no_tube_indices = torch.nonzero(out_put_tubes.cpu() == 0, as_tuple=True)[0]
    out_put_predict_no_tubes = out_put_predict[no_tube_indices]
    out_put_GT_no_tubes = out_put_GT[no_tube_indices]

    # Get patients without tubes and with pneumothorax
    pos_pt_ix = torch.nonzero(out_put_GT_no_tubes.cpu() == 1, as_tuple=True)[0]
    out_put_GT_no_tubes_pos_pt = out_put_GT_no_tubes[pos_pt_ix]
    out_put_predict_no_tubes_pos_pt = out_put_predict_no_tubes[pos_pt_ix]
    print(out_put_GT_no_tubes_pos_pt.size(), out_put_predict_no_tubes_pos_pt.size())

    out_put_GT_no_tubes_ds = torch.cat((out_put_GT_no_tubes_pos_pt, out_put_GT_neg_pt), dim=0)
    out_put_predict_no_tubes_ds = torch.cat((out_put_predict_no_tubes_pos_pt, out_put_predict_neg_pt), dim=0)
    print(out_put_GT_no_tubes_ds.size(), out_put_predict_no_tubes_ds.size())
    auroc_no_tubes, aurpc_no_tubes = compute_AUC(gt=out_put_GT_no_tubes_ds, pred=out_put_predict_no_tubes_ds)

    print(f"AuROC: {auroc_no_tubes}, AuRPC: {aurpc_no_tubes}")
    print(" ")

    torch.save(out_put_GT.cpu(), args.output_path / "GT.pth.tar")
    torch.save(out_put_predict.cpu(), args.output_path / "predictions.pth.tar")
    torch.save(out_put_tubes.cpu(), args.output_path / "tubes.pth.tar")
    torch.save(out_put_tube_prob.cpu(), args.output_path / "tube_prob.pth.tar")
    np.save(args.output_path / "image_names.npy", all_names)

    print(f"Outputs are saved at:{args.output_path}")

    df = pd.DataFrame({
        "Image_name": all_names,
        "GT": out_put_GT.squeeze().cpu().numpy().astype(int),
        "Predictions": torch.nn.Sigmoid()(out_put_predict.squeeze()).cpu().numpy(),
        "Tubes": out_put_tubes.squeeze().cpu().numpy().astype(int),
        "Predictions_bin": (torch.nn.Sigmoid()(out_put_predict.squeeze()).cpu().numpy() >= 0.5).astype(int)
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
    print(
        f"[Acc] overall pneumothorax: {overall_pneumothorax_acc}, "
        f"[Acc] pneumothorax with tube: {pneumothorax_w_tube_acc}, "
        f"[Acc] pneumothorax without tube: : {pneumothorax_wo_tube_acc}"
    )

    csv_file_path = args.output_path / "test_additional_info.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Saved additional info to {csv_file_path}")


def train(args, train_loader, valid_loader, optim, criterion):
    chk_pt = "chk_pt"
    metric = "metric"

    device = "cuda" if args.device == "cuda" else 'cpu'
    print(f'Using device: {device}')
    logger = SummaryWriter(args.tb_logs_path)

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    args.model.to(device)

    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = train_epoch(
            args=args,
            epoch=epoch,
            device=device,
            optimizer=optim,
            train_loader=train_loader,
            criterion=criterion,
            logger=logger
        )

        auc_valid, task_aucs, task_outputs, task_targets = valid_test_epoch(
            args=args,
            name='Valid',
            epoch=epoch,
            device=device,
            data_loader=valid_loader,
            criterion=criterion,
            logger=logger
        )

        if np.mean(auc_valid) > best_metric:
            try:
                os.remove(join(
                    args.chk_pt_path, f'{chk_pt}-best-auc{best_metric:4.4f}.pt')
                )  # remove previous best checkpoint
            except:
                pass
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = args.model.state_dict()
            torch.save(args.model, join(args.chk_pt_path, f'{chk_pt}-best-auc{np.mean(auc_valid):4.4f}.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric,
            'task_aucs': task_aucs[0],
            'task_recall': task_aucs[1],
        }

        metrics.append(stat)

        with open(join(args.output_path, f'{metric}-e{epoch + 1}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        if epoch % args.save_freq == 0:
            torch.save(args.model, join(args.chk_pt_path, f'{chk_pt}-e{epoch + 1}-auc{np.mean(auc_valid):4.4f}.pt'))

        # adjust_learning_rate(cfg, optim, epoch)

    return metrics, best_metric, weights_for_best_validauc


def train_epoch(args, epoch, device, train_loader, optimizer, criterion, limit=None, logger=None):
    args.model.train()

    if args.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights / weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)

    avg_loss = []
    t = tqdm(train_loader)
    num_batches = len(train_loader)

    for batch_idx, samples in enumerate(t):
        if limit and (batch_idx > limit):
            print("breaking out")
            break

        optimizer.zero_grad()
        try:
            images = samples["img"].float().to(device)
            targets = samples["lab"].to(device)
        except:
            images = samples[0].float().to(device)
            targets = samples[1].to(device)

        if len(targets.shape) == 1 and args.num_classes != 1:
            targets = F.one_hot(targets, num_classes=args.num_classes)

        if args.mixUp:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, args.alpha
            )
            images, targets_a, targets_b = map(Variable, (images,
                                                          targets_a, targets_b))
        if args.labelSmoothing:
            targets = targets * (1 - args.alpha) + args.alpha / targets.shape[0]
        outputs = args.model(images)

        loss = torch.zeros(1).to(device).float()
        for task in range(targets.shape[1]):

            task_output = outputs[:, task]
            task_target = targets[:, task]
            if len(task_target) > 0:
                if not (args.mixUp or args.focalLoss):
                    if len(args.pos_weights) > 0:
                        wt = torch.tensor([args.pos_weights[task]]).to('cuda')
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=wt)
                        task_loss = criterion(task_output.float(), task_target.float())
                    else:
                        task_loss = criterion(task_output.float(), task_target.float())
                    if torch.isnan(task_loss):
                        import pdb
                        pdb.set_trace()
                elif args.focalLoss:
                    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                    task_loss = criterion(task_output.float(), task_target.float())
                    pt = torch.exp(-task_loss)  # prevents nans when probability 0
                    focal_loss = 0.25 * (1 - pt) ** args.alpha * task_loss
                    task_loss = focal_loss.mean()
                elif args.mixUp:
                    task_targets_a = targets_a[:, task]
                    task_targets_b = targets_b[:, task]
                    task_loss = mixup_criterion(criterion, task_output.float(), task_targets_a.float(),
                                                task_targets_b.float(), lam)
                if args.taskweights:
                    loss += weights[task] * task_loss
                else:
                    loss += task_loss

        index = epoch * len(t) + batch_idx
        logger.add_scalar('train/task_loss', loss, index)
        logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], index)

        loss = loss.sum()
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        logger.add_scalar('train/total_loss', loss, index)
        if args.featurereg:
            feat = args.model.features(images)
            loss += feat.abs().sum()

        if args.weightreg:
            loss += args.model.classifier.weight.abs().sum()

        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

        if args.save_iters > 0 and epoch % args.save_freq == 0:
            idx_flag = int(num_batches / args.save_iters)
            if batch_idx % idx_flag == 0:
                torch.save(args.model, join(args.chk_pt_path, f'e{epoch + 1}-it{batch_idx}.pt'))
    return np.mean(avg_loss)


def valid_test_epoch(args, name, epoch, device, data_loader, criterion, limit=None, logger=None):
    args.model.eval()

    avg_loss = []
    task_outputs = []
    task_targets = []
    for task in range(args.num_classes):
        task_outputs.append([])
        task_targets.append([])

    with torch.no_grad():
        t = tqdm(data_loader)

        # iterate dataloader
        for batch_idx, samples in enumerate(t):
            index = epoch * len(t) + batch_idx
            if limit and (batch_idx > limit):
                print("breaking out")
                break
            try:
                images = samples["img"].to(device)
                targets = samples["lab"].to(device)
            except:
                images = samples[0].to(device)
                targets = samples[1].to(device)
            if len(targets.shape) == 1 and args.num_classes != 1:
                targets = F.one_hot(targets, num_classes=args.num_classes)
            outputs = args.model(images)
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:, task]
                task_target = targets[:, task]
                if len(task_target) > 0:
                    if len(args.pos_weights) > 0:
                        wt = torch.tensor([args.pos_weights[task]]).to('cuda')
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=wt)
                        loss += criterion(task_output.double(), task_target.double())
                    else:
                        loss += criterion(task_output.double(), task_target.double())
                    if torch.isnan(loss):
                        import pdb
                        pdb.set_trace()
                task_output = torch.sigmoid(task_output)
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            try:
                logger.add_scalar('valid/total_loss', loss, index)
            except:
                pass
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            logger.add_scalar('valid/avg_loss', np.mean(avg_loss), index)

        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])

        task_aucs = []
        task_recalls = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task])) > 1:
                try:
                    task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                except:
                    continue
                task_aucs.append(task_auc)
                y_pred = np.asarray(task_outputs[task] > 0.5).astype(int)
                task_recall = sklearn.metrics.recall_score(task_targets[task], y_pred)
                task_recalls.append(task_recall)

                cm = sklearn.metrics.confusion_matrix(task_targets[task], y_pred)
                print(cm, task, 0.5, "epoch: ", epoch)

                if not args.multi_class:  # its multi-label, calculate ece for each task
                    task_conf = (task_outputs[task] * y_pred) + (1 - y_pred) * (1 - task_outputs[task])
                    task_bin = uniform_binning(task_outputs[task], bin_size=0.10)
                    n_bins = np.max(task_bin) + 1
                    N = task_bin.shape[0]
                    ece = 0
                    for b in range(n_bins):
                        index = np.where(task_bin == b)
                        y_pred_b = y_pred[index]
                        y_true_b = task_targets[task][index]
                        y_conf_b = task_conf[index]
                        if y_pred_b.shape[0] == 0:
                            ece += 0
                        else:
                            acc = np.sum(y_pred_b == y_true_b) / y_pred_b.shape[0]
                            c = y_pred_b.shape[0]
                            conf = np.mean(y_conf_b)
                            ece += np.abs(acc - conf) * (c / N)
                    logger.add_scalar('valid/ece_task_' + str(task), ece, epoch)
                    print('ece task ' + str(task) + ' : ', ece)
            else:
                task_aucs.append(np.nan)
                task_recalls.append(np.nan)
        if args.multi_class:
            task_outputs = np.asarray(task_outputs)
            task_targets = np.asarray(task_targets)
            task_outputs = np.transpose(np.asarray(task_outputs))
            task_targets = np.transpose(np.asarray(task_targets))
            y_pred = np.argmax(task_outputs, axis=1)
            y_conf = np.max(task_outputs, axis=1)
            y_true = np.argmax(task_targets, axis=1)
            y_bin = uniform_binning(y_conf, bin_size=0.10)
            n_bins = np.max(y_bin) + 1
            N = y_bin.shape[0]
            ece = 0
            for b in range(n_bins):
                index = np.where(y_bin == b)
                y_pred_b = y_pred[index]
                y_true_b = y_true[index]
                y_conf_b = y_conf[index]
                if y_pred_b.shape[0] == 0:
                    ece += 0
                else:
                    acc = np.sum(y_pred_b == y_true_b) / y_pred_b.shape[0]
                    c = y_pred_b.shape[0]
                    conf = np.mean(y_conf_b)
                    ece += np.abs(acc - conf) * (c / N)
            logger.add_scalar('valid/ece', ece, epoch)
            print('ece: ', ece)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    task_recalls = np.asarray(task_recalls)
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')
    print("Tasks AUC:")
    print(task_aucs)
    print("Tasks Recall:")
    print(task_recalls)
    return auc, [task_aucs, task_recalls], task_outputs, task_targets


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
