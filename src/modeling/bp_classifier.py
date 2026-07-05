from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass(frozen=True)
class _TensorBatch:
    features: torch.Tensor
    target: torch.Tensor
    sample_weight: torch.Tensor


class BPNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """A small BP neural network classifier compatible with sklearn Pipeline."""

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.10,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        max_epochs: int = 180,
        batch_size: int = 128,
        patience: int = 20,
        positive_weight: float = 1.0,
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.positive_weight = positive_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.random_state = random_state

    def fit(self, x, y, sample_weight=None):
        self._set_seed()
        x_array = self._to_feature_array(x)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        if sample_weight is None:
            weight_array = np.ones(len(y_array), dtype=np.float32)
        else:
            weight_array = np.asarray(sample_weight, dtype=np.float32).reshape(-1)

        classes = np.unique(y_array.astype(int))
        if len(classes) < 2:
            raise ValueError("BPNeuralNetworkClassifier requires two classes.")
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = x_array.shape[1]
        self.model_ = self._build_model(self.n_features_in_)

        dataset = _TensorBatch(
            features=torch.as_tensor(x_array, dtype=torch.float32),
            target=torch.as_tensor(y_array, dtype=torch.float32),
            sample_weight=torch.as_tensor(weight_array, dtype=torch.float32),
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        positive_weight = torch.tensor(float(max(self.positive_weight, 1e-6)), dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")

        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(int(self.max_epochs)):
            epoch_loss = self._train_epoch(dataset, optimizer, criterion)
            if epoch_loss + 1e-7 < best_loss:
                best_loss = epoch_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= int(self.patience):
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x):
        self._check_fitted()
        x_array = self._to_feature_array(x)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(x_array, dtype=torch.float32)).reshape(-1)
            positive = torch.sigmoid(logits).cpu().numpy()
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, x):
        proba = self.predict_proba(x)[:, 1]
        return (proba >= 0.5).astype(int)

    def _build_model(self, input_dim: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in self.hidden_layers:
            layers.append(torch.nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(torch.nn.LayerNorm(int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            if float(self.dropout) > 0:
                layers.append(torch.nn.Dropout(float(self.dropout)))
            previous_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(previous_dim, 1))
        return torch.nn.Sequential(*layers)

    def _train_epoch(
        self,
        dataset: _TensorBatch,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        self.model_.train()
        sample_count = len(dataset.target)
        indexes = torch.randperm(sample_count)
        batch_size = max(1, int(self.batch_size))
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, sample_count, batch_size):
            batch_index = indexes[start : start + batch_size]
            features = dataset.features[batch_index]
            target = dataset.target[batch_index]
            sample_weight = dataset.sample_weight[batch_index]

            optimizer.zero_grad()
            logits = self.model_(features).reshape(-1)
            loss = criterion(logits, target)
            gamma = float(self.focal_gamma)
            if gamma > 0:
                proba = torch.sigmoid(logits).detach()
                pt = torch.where(target > 0.5, proba, 1.0 - proba).clamp(1e-6, 1.0 - 1e-6)
                alpha = float(self.focal_alpha)
                alpha_t = torch.where(
                    target > 0.5,
                    torch.full_like(target, alpha),
                    torch.full_like(target, 1.0 - alpha),
                )
                loss = loss * alpha_t * torch.pow(1.0 - pt, gamma)
            weighted_loss = (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            weighted_loss.backward()
            optimizer.step()

            batch_weight = float(sample_weight.sum().item())
            total_loss += float(weighted_loss.item()) * batch_weight
            total_weight += batch_weight

        return total_loss / max(total_weight, 1e-6)

    def _set_seed(self) -> None:
        seed = int(self.random_state)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

    @staticmethod
    def _to_feature_array(x) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise ValueError("BPNeuralNetworkClassifier is not fitted yet.")
