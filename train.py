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
from utils.utils import load_pretrained
# from torchsummary import summary

from data.datamodule import S3DataLoader, S3BatchDataset
from backbone.transformer import TwoWayTransformer as transformer
from models.WebSAMAdapter import WebSAMEncoder, WebSAMDecoder, WebSAMAdapter


logging.basicConfig(filename='training.log', level=logging.INFO)


# Define custom loss function
class BCEIoULoss(nn.Module):
    def __init__(self):
        super(BCEIoULoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou_loss = 1 - (intersection + 1) / (union + 1)
        return bce_loss + iou_loss.mean()

def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, sam_path, num_epochs=20, save_dir='./saved_models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    load_pretrained(model, sam_path, ignore=['output_upscaling'])
    model.train()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in progress_bar:
            images, ground_truths = zip(*batch)
            images = torch.stack(images).to(device).squeeze(1)
            ground_truths = torch.stack(ground_truths).to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images).squeeze(0)
            # Calculate loss
            # pdb.set_trace()
            loss = criterion(outputs, ground_truths)
            # Backward pass and optimize
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

        # Save model weights every other epoch
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model weights at epoch {epoch + 1} to {save_path}")
            logging.info(f'Saved model weights at epoch {epoch + 1} to {save_path}')

    print("Training completed.")
    logging.info('Training completed.')

def validate_model(val_dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            images, ground_truths = zip(*batch)
            images = torch.stack(images).to(device).squeeze(1)
            ground_truths = torch.stack(ground_truths).to(device)
            outputs = model(images).squeeze(0)
            loss = criterion(outputs, ground_truths)
            val_loss += loss.item()
    return val_loss / len(val_dataloader)

if __name__ == "__main__":
    load_dotenv()
    bucket_name = "webis-webseg20"
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    data_loader = S3DataLoader(
        bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    file_path = 'data/indices.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()

    indices = [line.strip() for line in lines if line.strip()][:-1]
    
    train_size = int(0.7 * len(indices))
    val_size = int(0.2 * len(indices))
    test_size = len(indices) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(indices, [train_size, val_size, test_size])

    train_indices = [indices[i] for i in train_indices.indices]
    val_indices = [indices[i] for i in val_indices.indices]
    test_indices = [indices[i] for i in test_indices.indices]

    batch_size = 2
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    train_dataset = S3BatchDataset(data_loader, train_indices, batch_size, transform=resize_transform)
    val_dataset = S3BatchDataset(data_loader, val_indices, batch_size, transform=resize_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize model, loss, and optimizer
    encoder = WebSAMEncoder()
    twt = transformer(depth = 12, embedding_dim = 256, num_heads = 8, mlp_dim = 256)
    decoder = WebSAMDecoder(transformer_dim = 256, transformer = twt)

    model = WebSAMAdapter(encoder, decoder)
    # summary(model, input_size=(3, 1024, 1024))

    criterion = BCEIoULoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    # Train the model
    sam_path = 'sam_b.pt'  # Add the correct path to SAM weights here
    train_model(train_dataloader, val_dataloader, model, criterion, optimizer, sam_path, num_epochs=20)
