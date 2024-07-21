import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import load_pretrained
from data.datamodule import LocalDataLoader, LocalBatchDataset
from backbone.transformer import TwoWayTransformer as transformer
from models.WebSAMAdapter import WebSAMEncoder, WebSAMDecoder, WebSAMAdapter

logging.basicConfig(filename='training.log', level=logging.INFO)

class BCEIoULoss(nn.Module):
    def __init__(self):
        super(BCEIoULoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        inputs = torch.sigmoid(inputs).round()
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        iou_loss = 1 - (intersection + 1) / (union + 1)
        iou_loss = iou_loss.mean()
        print("BCE Loss: ", bce_loss)
        print("IOU Loss: ", iou_loss)
        print(f"Inputs requires grad: {inputs.requires_grad}")
        print(f"Targets requires grad: {targets.requires_grad}")
        return bce_loss + iou_loss


def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, batch_size, num_epochs=20, save_dir='./saved_models', start_epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("CUDA available: ", torch.cuda.is_available())
    print("device: ", device)
    model.train()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in progress_bar:
            image_tuples, targets = zip(*batch)
            images, image_shapes = zip(*image_tuples)
            # to parallelize the encoder-decoder passes
            images = torch.stack(images).to(device).squeeze(1)
            image_shapes = torch.stack(image_shapes).to(device)
            print(f"GPU Memory Usage (current, max): {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB")
            optimizer.zero_grad()
            outputs = model(images, image_shapes) # List of length batch_size containing each images corresponding mask

            loss = 0
            for output, target in zip(outputs, targets):
                loss += criterion(output, target.unsqueeze(0).to(device))
        
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        logging.info(f'Epoch: {epoch+1}, Training Loss: {epoch_loss:.4f}')

        val_loss = validate_model(val_dataloader, model, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        logging.info(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}')

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, save_path)
            print(f"Saved model weights at epoch {epoch + 1} to {save_path}")
            logging.info(f'Saved model weights at epoch {epoch + 1} to {save_path}')

    print("Training completed.")
    logging.info('Training completed.')

def validate_model(val_dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            image_tuples, targets = zip(*batch)
            images, image_shapes = zip(*image_tuples)
            images = torch.stack(images).to(device).squeeze(1)
            image_shapes = torch.stack(image_shapes).to(device)
            print(f"GPU Memory Usage (current, max): {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB")
            optimizer.zero_grad()
            outputs = model(images, image_shapes) 

            loss = 0
            for output, target in zip(outputs, targets):
                loss += criterion(output, target.unsqueeze(0).to(device))
        
            val_loss += loss.item()
    return val_loss / len(val_dataloader)

if __name__ == "__main__":
    base_path = "/shared_data/mlr_club/3988124"
    
    data_loader = LocalDataLoader(base_path)

    contents = data_loader.ls('')
    print("Directories in base path:", contents)

    file_path = 'data/indices.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()

    indices = [line.strip() for line in lines if line.strip()][:-1]
    
    seed = 42
    torch.manual_seed(seed)

    train_size = int(0.7 * len(indices))
    val_size = int(0.2 * len(indices))
    test_size = len(indices) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(indices, [train_size, val_size, test_size], generator=generator)

    train_indices = [indices[i] for i in train_indices.indices]
    val_indices = [indices[i] for i in val_indices.indices]
    test_indices = [indices[i] for i in test_indices.indices]

    batch_size = 8
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    train_dataset = LocalBatchDataset(data_loader, train_indices, batch_size, transform=resize_transform)
    val_dataset = LocalBatchDataset(data_loader, val_indices, batch_size, transform=resize_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    encoder = WebSAMEncoder()
    twt = transformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
    decoder = WebSAMDecoder(transformer_dim=256, transformer=twt)

    model = WebSAMAdapter(encoder, decoder)

    criterion = BCEIoULoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    start_epoch = 0
    checkpoint_path = './saved_models'
    latest_checkpoint = None
    # if os.path.exists(checkpoint_path):
    #     checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
    #     if checkpoint_files:
    #         latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # if latest_checkpoint:
    #     checkpoint = torch.load(os.path.join(checkpoint_path, latest_checkpoint))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     print(f"Resuming training from epoch {start_epoch + 1}")
    # else:

    sam_path = 'checkpoints/sam_b.pt'
    model = load_pretrained(model, sam_path, ignore=[])

    train_model(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, batch_size, num_epochs=20, start_epoch=start_epoch)
