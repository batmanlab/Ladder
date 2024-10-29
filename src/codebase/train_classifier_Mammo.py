import warnings
from pathlib import Path

import torch

from Classifiers.experiments_RSNA import do_experiments
from utils import get_Paths, seed_all

warnings.filterwarnings("ignore")
import argparse
import os
import pickle


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0',
                        help='path to checkpoints')
    parser.add_argument('--output_path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0',
                        help='path to output logs')
    parser.add_argument(
        "--data-dir",
        default="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/",
        type=str, help="Path to data file"
    )
    parser.add_argument(
        "--img-dir", default="RSNA_Cancer_Detection/train_images_png", type=str, help="Path to image file"
    )
    parser.add_argument("--csv-file", default="", type=str,
                        help="data csv file")
    parser.add_argument("--detector-threshold", default=0.1, type=float)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)
    parser.add_argument("--dataset", default="RSNA_breast", type=str)
    parser.add_argument("--data_frac", default=1.0, type=float)
    parser.add_argument("--label", default="cancer", type=str)
    parser.add_argument("--VER", default="084", type=str)
    parser.add_argument("--arch", default="tf_efficientnet_b5_ns", type=str)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--focal-alpha", default=0.6, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument("--inference-mode", default='n', type=str)
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--weighted-BCE", default='n', type=str)
    parser.add_argument("--balanced-dataloader", default='n', type=str)

    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    # get paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.apex = True if args.apex == "y" else False
    args.pretrained_swin_encoder = True if args.pretrained_swin_encoder == "y" else False
    args.swin_model_type = True if args.swin_model_type == "y" else False
    args.running_interactive = True if args.running_interactive == "y" else False

    args.chk_pt_path = Path(args.checkpoints)
    args.output_path = Path(args.output_path)
    args.tb_logs_path = Path(args.tensorboard_path)

    os.makedirs(args.chk_pt_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.tb_logs_path, exist_ok=True)
    print("====================> Paths <====================")
    print(f"checkpoint_path: {args.chk_pt_path}")
    print(f"output_path: {args.output_path}")
    print(f"tb_logs_path: {args.tb_logs_path}")
    print('device:', device)
    print('torch version:', torch.__version__)
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(args.output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()

    if args.weighted_BCE == "y" and args.dataset.lower() == "rsna" and args.label.lower() == "cancer":
        args.BCE_weights = {
            "fold0": 46.48148148148148,
            "fold1": 46.01830663615561,
            "fold2": 46.41339491916859,
            "fold3": 46.05747126436781
        }
    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "cancer":
        args.BCE_weights = {
            "fold0": 12.756594724220623,
            "fold1": 12.756594724220623,
            "fold2": 12.756594724220623,
            "fold3": 12.756594724220623
        }
    do_experiments(args, device)


if __name__ == "__main__":
    args = config()
    if torch.cuda.is_available():
        print("CUDA is available on this system.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available on this system.")

    main(args)
