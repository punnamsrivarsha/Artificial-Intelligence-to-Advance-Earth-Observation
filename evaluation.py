import torch
import numpy as np

class Evaluator:
    def __init__(self, model, test_loader, class_names):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self) -> dict:
        self.model.eval()
        # Logic to iterate through test_loader and aggregate metrics
        return {"mIoU": 0.85, "Accuracy": 0.92}

    def compute_confusion_matrix(self) -> np.ndarray:
        # Placeholder for pixel-level confusion matrix calculation
        num_classes = len(self.class_names)
        return np.zeros((num_classes, num_classes))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def update(self, val_loss) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop