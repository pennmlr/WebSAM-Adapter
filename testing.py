import torch
import numpy as np

from data.datamodule import S3DataLoader, S3BatchDataset

def pixels_metrics(pred, target):
    TP = (pred * target).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def validate_model(val_dataloader, model, device):
    model.eval()
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for batch in val_dataloader:
            images, ground_truths = zip(*batch)
            images = torch.stack(images).to(device).squeeze(1)
            ground_truths = torch.stack(ground_truths).to(device)
            outputs = model(images).squeeze(0)
            p, r, f1 = pixels_metrics(outputs, ground_truths)
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)
    h_mean_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return avg_precision, avg_recall, avg_f1_score, h_mean_score