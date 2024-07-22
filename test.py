import os
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from data.datamodule import LocalDataLoader, LocalBatchDataset, LocalBatchEvalDataset

ELEMENTS = ['pixels', 'fine_edges', 'coarse_edges', 'dom_nodes']

def evaluation_metric(pred, target, mask):
    TP = (pred * target * mask).sum().item()
    FP = (((pred == 1) & (target == 0)) * mask).sum().item()
    FN = (((pred == 0) & (target == 1)) * mask).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def validate_model(val_dataloader, model, device, elements=ELEMENTS):
    model.eval()

    scores = {}
    for elt in elements:
        precisions = []
        recalls = []
        f1_scores = []

        with torch.no_grad():
            for batch in val_dataloader:
                print('in batch')
                image_tuples, targets, fine_edges_masks, coarse_edges_masks, dom_nodes_masks = zip(*batch)
                images, image_shapes = zip(*image_tuples)
                images = torch.stack(images).to(device).squeeze(1)
                image_shapes = torch.stack(image_shapes).to(device)
                targets = torch.stack(targets).to(device)
                outputs = model(images).squeeze(0)
                # outputs = torch.randint(0, 2, (1024, 1024), dtype=torch.float32)

                masks = [torch.ones(image_shapes, dtype=torch.float32), fine_edges_masks, coarse_edges_masks, dom_nodes_masks]
                p, r, f1 = evaluation_metric(outputs, targets, masks[ELEMENTS.index(elt)])
                precisions.append(p)
                recalls.append(r)
                f1_scores.append(f1)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1_score = np.mean(f1_scores)
        h_mean_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        scores[elt] = {'precision': avg_precision, 'recall': avg_recall, 'f1-score': avg_f1_score, 'h-mean-score': h_mean_score}
    
    return scores


if __name__ == "__main__":
    base_path = "/shared_data/mlr_club/3988124"
    
    data_loader = LocalDataLoader(base_path)

    contents = data_loader.ls('')
    print("Directories in base path:", contents)

    file_path = 'data/indices.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()

    indices = [line.strip() for line in lines if line.strip()][:-1]

    batch_size = 1
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    dataset = LocalBatchEvalDataset(data_loader, indices, batch_size, transform=resize_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('begin validating model')
    scores = validate_model(val_dataloader=dataloader, model=None, device=device)
    print(scores)