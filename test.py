import os
import pdb
from collections import defaultdict
import torch
import numpy as np

from utils.utils import load_pretrained

from torchvision import transforms
from torch.utils.data import DataLoader
from data.datamodule import LocalDataLoader, LocalBatchDataset, LocalBatchEvalDataset
from backbone.transformer import TwoWayTransformer as transformer
from models.WebSAMAdapter import WebSAMEncoder, WebSAMDecoder, WebSAMAdapter

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

    precisions = defaultdict(list)
    recalls = defaultdict(list)
    f1_scores = defaultdict(list)
    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            print(f"in batch: {count}")
            image_tuples, targets, fine_edges_masks, coarse_edges_masks, dom_nodes_masks = zip(*batch)
            images, original_image_shapes = zip(*image_tuples)
            images = torch.stack(images).to(device).squeeze(1)
            image_shapes = torch.stack(original_image_shapes).to(device)
            outputs = model(images, image_shapes)

            for output, target, fine_edges_mask, coarse_edges_mask, dom_nodes_mask in zip(outputs, targets, fine_edges_masks, coarse_edges_masks, dom_nodes_masks):
                output = torch.sigmoid(output).squeeze(0)
                output_binary = output.cpu().numpy().round()
                full_mask = torch.ones(output.shape, dtype=torch.float32)
                masks = [full_mask, fine_edges_mask, coarse_edges_mask, dom_nodes_mask]
                for index, elt in enumerate(elements):
                    p, r, f1 = evaluation_metric(output_binary, target, masks[index])
                    precisions[elt].append(p)
                    recalls[elt].append(r)
                    f1_scores[elt].append(f1)

    scores = {}
    for elt in elements:
        avg_precision = np.mean(precisions[elt])
        avg_recall = np.mean(recalls[elt])
        avg_f1_score = np.mean(f1_scores[elt])
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

    batch_size = 2
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    dataset = LocalBatchEvalDataset(data_loader, indices, batch_size, transform=resize_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = WebSAMEncoder()
    twt = transformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
    decoder = WebSAMDecoder(transformer_dim=256, transformer=twt)
    model = WebSAMAdapter(encoder, decoder)

    checkpoint_path = '/shared_data/mlr_club/saved_models/model_epoch_12.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print('begin validating model')
    scores = validate_model(val_dataloader=dataloader, model=model, device=device)
    print(scores)
