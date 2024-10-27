import warnings
from pathlib import Path

import torch

from Classifiers.experiments_CXR import train
from med_img_datasets_clf.dataset_utils import get_dataset
from utils import seed_all
from Classifiers.models.ResNet import ResNet
warnings.filterwarnings("ignore")
import argparse
import os
import pickle

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/log/NIH/seed{}',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}',
                        help='path to checkpoints')
    parser.add_argument('--output_path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}',
                        help='path to output logs')
    parser.add_argument(
        "--data-file",
        default="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/nih/nih_processed_v2.csv",
        type=str, help="Path to nih data file"
    )
    parser.add_argument(
        "--column-name-split", default="val_train_split", type=str,
        help="name of the column using which we split the data"
    )
    parser.add_argument(
        "--pos_weights", nargs='+', default=[20.16], help="postive weights for Pneumothorax"
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="hyper-parameter for mix-up and label smoothing")
    parser.add_argument("--class-names", default="Pneumothorax", type=str, help="disease to predict")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--save-freq", default=1, type=int)
    parser.add_argument("--save-iters", default=0, type=int)
    parser.add_argument("--num-epochs", default=60, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--channel", default=1, type=int)
    parser.add_argument("--imgpath", default="", type=str)
    parser.add_argument("--dataset", default="NIH", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--arch", default="densenet121", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument('--mixUp', default="n", type=str)
    parser.add_argument('--labelSmoothing', default="n", type=str)
    parser.add_argument('--focalLoss', default="n", type=str)
    parser.add_argument('--taskweights', default="y", type=str)
    parser.add_argument('--featurereg', default="n", type=str)
    parser.add_argument('--weightreg', default="n", type=str)
    parser.add_argument('--multi-class', default="n", type=str)
    parser.add_argument('--loss', default="BCE_W", type=str)
    parser.add_argument('--TS', default="n", type=str)
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--drop-rate", default=0.00, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--data-aug-hf", default=0.5, type=float)
    parser.add_argument("--partition-name", nargs='+', default=["test"])

    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    # get paths
    chk_pt_path = Path(args.checkpoints.format(args.seed))
    output_path = Path(args.output_path.format(args.seed))
    tb_logs_path = Path(args.tensorboard_path.format(args.seed))
    os.makedirs(chk_pt_path, exist_ok=True)
    # os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("====================> Paths <====================")
    print(f"checkpoint_path: {chk_pt_path}")
    print(f"output_path: {output_path}")
    print(f"tb_logs_path: {tb_logs_path}")
    print("====================> Paths <====================")

    args.mixUp = False if args.mixUp == "n" else True
    args.labelSmoothing = False if args.labelSmoothing == "n" else True
    args.focalLoss = False if args.focalLoss == "n" else True
    args.taskweights = False if args.taskweights == "n" else True
    args.featurereg = False if args.featurereg == "n" else True
    args.weightreg = False if args.weightreg == "n" else True
    args.multi_class = False if args.multi_class == "n" else True
    args.TS = False if args.TS == "n" else True
    args.class_names = [args.class_names]
    args.chk_pt_path = chk_pt_path
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path
    args.weights = None
    args.return_logit = False

    pickle.dump(args, open(os.path.join(output_path, "NIH_train_classifier_configs.pkl"), "wb"))

    # get dataset
    train_loader, valid_loader = get_dataset(args)

    # get models and train configs
    args.model = ResNet(
        pre_trained=True, n_class=args.num_classes, model_choice=args.arch, weights=args.weights,
        return_logit=args.return_logit
    )
    optimizer = torch.optim.Adam(args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    pos_wt = torch.tensor([args.pos_weights[0]]).to('cuda')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)

    train(args, train_loader, valid_loader, optimizer, criterion)


if __name__ == "__main__":
    args = config()
    main(args)
