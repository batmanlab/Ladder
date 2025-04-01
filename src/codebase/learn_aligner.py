import copy
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from utils import seed_all

warnings.filterwarnings("ignore")
import argparse
import os
import logging


class LinearAligner:
    """
        A linear projection model that learns to align classifier embeddings to CLIP (VLM) embeddings.
    """

    def __init__(self) -> None:
        self.W = None
        self.b = None

    def train(self, reps_clf_train, reps_clip_train, reps_clf_test, reps_clip_test, lr=0.01, epochs=10,
              target_variance=4.5, verbose=0) -> dict:
        """
                Trains the linear projection model using paired representations from classifier and CLIP.

                Args:
                    reps_clf_train (np.ndarray): Classifier embeddings (train set).
                    reps_clip_train (np.ndarray): CLIP embeddings (train set).
                    reps_clf_test (np.ndarray): Classifier embeddings (validation set).
                    reps_clip_test (np.ndarray): CLIP embeddings (validation set).
                    lr (float): Learning rate.
                    epochs (int): Number of training epochs.
                    target_variance (float): Variance scaling target.
                    verbose (int): Verbosity level.

                Returns:
                    dict: Dictionary containing trained weights and bias.
        """
        lr_solver = LinearRegressionSolver()

        print(f'Training linear aligner ...')
        print(f'Linear alignment train: ({reps_clf_train.shape}) --> ({reps_clip_train.shape}).')
        print(f'Linear alignment test: ({reps_clf_test.shape}) --> ({reps_clip_test.shape}).')

        logging.info(f'Training linear aligner ...')
        logging.info(f'Linear alignment train: ({reps_clf_train.shape}) --> ({reps_clip_train.shape}).')
        logging.info(f'Linear alignment test: ({reps_clf_test.shape}) --> ({reps_clip_test.shape}).')

        var1 = lr_solver.get_variance(reps_clf_train)
        var2 = lr_solver.get_variance(reps_clip_train)

        c1 = (target_variance / var1) ** 0.5
        c2 = (target_variance / var2) ** 0.5

        reps_clf_train = c1 * reps_clf_train
        reps_clip_train = c2 * reps_clip_train
        reps_clf_test = c1 * reps_clf_test
        reps_clip_test = c2 * reps_clip_test

        W, b = lr_solver.train(
            reps_clf_train, reps_clip_train, reps_clf_test, reps_clip_test, bias=True, epochs=epochs, lr=lr,
            batch_size=100)

        W = W * c1 / c2
        b = b * c1 / c2

        self.W = W
        self.b = b

    def get_aligned_representation(self, ftrs):
        """
                Projects classifier features into the aligned space.

                Args:
                    ftrs (torch.Tensor or np.ndarray): Classifier features.

                Returns:
                    torch.Tensor: Projected features in aligned space.
        """
        return ftrs @ self.W.T + self.b

    def load_W(self, path_to_load: str):
        """
                Loads aligner weights from file.

                Args:
                    path_to_load (str): Path to saved model weights.
        """
        aligner_dict = torch.load(path_to_load)
        self.W, self.b = [aligner_dict[x].float() for x in ['W', 'b']]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W = self.W.to(device).float()
        self.b = self.b.to(device).float()

    def save_W(self, path_to_save):
        """
                Saves the aligner weights to disk.

                Args:
                    path_to_save (Path): Path to save the weights (.pth).
        """
        torch.save({'b': self.b.detach().cpu(), 'W': self.W.detach().cpu()}, path_to_save)
        print(f'Aligner weights saved to {path_to_save}')
        logging.info(f'Aligner weights saved to {path_to_save}')


class LinearRegression(torch.nn.Module):
    """
        Simple linear regression model using PyTorch.
    """

    def __init__(self, input_size, output_size, bias=True):
        """
                Args:
                    input_size (int): Input feature size.
                    output_size (int): Output feature size.
                    bias (bool): Whether to include a bias term.
        """
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out


class LinearRegressionSolver:
    """
        Solver class for training and evaluating linear regression.
    """

    def __init__(self):
        self.model = None
        self.criterion = torch.nn.MSELoss()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
              lr=0.01, bias=True, batch_size=100, epochs=20):
        """
                Trains a linear regression model on training data and selects the best weights by validation MSE.

                Args:
                    X_train (np.ndarray): Input features for training.
                    y_train (np.ndarray): Target features for training.
                    X_test (np.ndarray): Input features for validation.
                    y_test (np.ndarray): Target features for validation.
                    lr (float): Learning rate.
                    bias (bool): Include bias term.
                    batch_size (int): Training batch size.
                    epochs (int): Number of epochs.

                Returns:
                    tuple: Best weight and bias tensors.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tensor_X_train = torch.from_numpy(X_train).float()
        tensor_y_train = torch.from_numpy(y_train).float()
        dataset_train = torch.utils.data.TensorDataset(tensor_X_train, tensor_y_train)
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)

        self.model = LinearRegression(X_train.shape[1], y_train.shape[1], bias=bias)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        self.model.to(device)

        init_mse, init_r2 = self.test(X_test, y_test)
        print(f'Initial MSE, R^2: {init_mse:.3f}, {init_r2:.3f}')
        logging.info(f'Initial MSE, R^2: {init_mse:.3f}, {init_r2:.3f}')

        self.init_result = init_r2
        self.model.train()
        best_mse = 99999999
        best_W = 0
        best_b = 0
        for epoch in range(epochs):
            e_loss, num_of_batches = 0, 0

            for batch_idx, (inputs, targets) in enumerate(dataloader_train):
                num_of_batches += 1
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                e_loss += loss.item()

                loss.backward()
                optimizer.step()

            e_loss /= num_of_batches
            epoch_test_mse, epoch_test_r2 = self.test(X_test, y_test)

            if epoch_test_mse < best_mse:
                best_mse = epoch_test_mse
                for name, param in self.model.named_parameters():
                    if name == 'linear.weight':
                        best_W = copy.deepcopy(param.detach())
                    else:
                        best_b = copy.deepcopy(param.detach())

                print(f" best W: {best_W[0][0:10]}, best b: {best_b[0:10]}")
            print(
                f'Epoch number, {epoch}, train loss: {e_loss:.3f}, '
                f'test MSE: {epoch_test_mse:.3f}, test_r2: {epoch_test_r2:.3f}, '
                f'best MSE: {best_mse:.3f}'
            )
            logging.info(
                f'Epoch number, {epoch}, train loss: {e_loss:.3f}, '
                f'test MSE: {epoch_test_mse:.3f}, test_r2: {epoch_test_r2:.3f}, '
                f'best MSE: {best_mse:.3f}'
            )

            scheduler.step()

        return best_W, best_b

    def get_variance(self, y: np.ndarray):
        """
                Computes variance of input array.

                Args:
                    y (np.ndarray): Input array.

                Returns:
                    float: Variance value.
        """
        ey = np.mean(y)
        ey2 = np.mean(np.square(y))
        return ey2 - ey ** 2

    def test(self, X: np.ndarray, y: np.ndarray, batch_size=100):
        """
                Evaluates the model on test data using MSE and R^2.

                Args:
                    X (np.ndarray): Test features.
                    y (np.ndarray): Ground truth.
                    batch_size (int): Batch size.

                Returns:
                    tuple: (MSE, R-squared score)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tensor_X = torch.from_numpy(X).float()
        tensor_y = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        self.model.eval()

        total_mse_err, num_of_batches = 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                num_of_batches += 1
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_mse_err += loss.item()

        total_mse_err /= num_of_batches

        return total_mse_err, 1 - total_mse_err / self.get_variance(y)


def init_path(seed, clf_reps_path, clip_reps_path, save_path, dataset):
    """
        Loads classifier and CLIP representations for training and validation folds.

        Args:
            seed (int): Random seed.
            clf_reps_path (str): Path to classifier embeddings with `{seed}`, `{split}` placeholders.
            clip_reps_path (str): Path to CLIP embeddings with `{seed}`, `{split}` placeholders.
            save_path (str): Directory path to save outputs (with `{seed}` placeholder).
            dataset (str): Dataset name (not used but kept for compatibility).

        Returns:
            tuple: Classifier/CLIP embeddings for train/test and formatted save path.
    """
    reps_clf_train = np.load(clf_reps_path.format(seed, "train"))
    reps_clip_train = np.load(clip_reps_path.format(seed, "train"))
    reps_clf_test = np.load(clf_reps_path.format(seed, "valid"))
    reps_clip_test = np.load(clip_reps_path.format(seed, "valid"))
    save_path = Path(save_path.format(seed))
    return reps_clf_train, reps_clip_train, reps_clf_test, reps_clip_test, save_path


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NIH", type=str)
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/aligner",
        help='save path of aligners')
    parser.add_argument(
        "--clf_reps_path",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_RN50/{1}_classifier_embeddings.npy",
        type=str, help="Save path of classifier representations"
    )
    parser.add_argument(
        "--clip_reps_path",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_RN50/{1}_clip_embeddings.npy",
        type=str,
        help="Save path of clip (VLM) representations"
    )

    parser.add_argument("--fold", default=0, type=int, help="which fold?")
    parser.add_argument("--seed", default=0, type=int, help="which seed?")
    parser.add_argument("--epochs", default=50, type=int, help="Epochs to train?")
    parser.add_argument("--lr", default=0.01, type=float, help="Epochs to train?")
    return parser.parse_args()


def main(args):
    seed_all(args.seed)

    reps_clf_train, reps_clip_train, reps_clf_test, reps_clip_test, save_path = init_path(
        args.seed, args.clf_reps_path, args.clip_reps_path, args.save_path, args.dataset
    )
    print(save_path / "aligner_out.txt")
    logging.basicConfig(
        filename=save_path / "aligner_out.txt", level=logging.INFO, format='%(asctime)s - %(message)s')

    print("\n")
    print(f"Train size: classifier [{reps_clf_train.shape}], clip [{reps_clip_train.shape}]")
    print(f"Valid size: classifier [{reps_clf_test.shape}], clip [{reps_clip_test.shape}]")
    logging.info(f"Train size: classifier [{reps_clf_train.shape}], clip [{reps_clip_train.shape}]")
    logging.info(f"Valid size: classifier [{reps_clf_test.shape}], clip [{reps_clip_test.shape}]")

    linear_aligner = LinearAligner()
    linear_aligner.train(
        reps_clf_train, reps_clip_train, reps_clf_test, reps_clip_test, epochs=args.epochs, lr=args.lr,
        target_variance=4.5)

    os.makedirs(save_path, exist_ok=True)
    linear_aligner.save_W(save_path / f"aligner_{args.epochs}.pth")
    print(f"Saved aligner to {save_path}")
    logging.info(f"Saved aligner to {save_path}")


if __name__ == "__main__":
    args = config()
    main(args)
