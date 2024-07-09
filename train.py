import os
import torch
import torch.nn as nn
import torch.optim as optim

from dotenv import load_dotenv
from torchvision import transforms
from torch.utils.data import DataLoader

from models.WebSAMAdapter import WebSAMEncoder, WebSAMDecoder, WebSAMAdapter
from backbone.transformer import TwoWayTransformer as transformer
from data.datamodule import S3DataLoader, S3BatchDataset

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
    
def train_model(train_dataloader, model, criterion, optimizer, num_epochs=20, save_dir='./saved_models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            image, ground_truth = batch
            image = image.to(device)
            ground_truth = ground_truth.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(image)
            # Calculate loss
            loss = criterion(outputs, ground_truth)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        # Save model weights every other epoch
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model weights at epoch {epoch + 1} to {save_path}")

    print("Training completed.")

if __name__ == "__main__":
    #remove later?
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

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

    indices = [line.strip() for line in lines if line.strip()]
    train_indices, test_indices = torch.utils.data.random_split(indices, [int(0.7 * len(indices)), int(0.3 * len(indices))])
    batch_size = 5
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    train_dataset = S3BatchDataset(data_loader, train_indices, batch_size, transform=resize_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Initialize model, loss, and optimizer
    #TODO: add SAM weight loader call
    encoder = WebSAMEncoder()
    twt = transformer(depth = 12, embedding_dim = 256, num_heads = 8, mlp_dim = 256)
    decoder = WebSAMDecoder(transformer_dim = 256, transformer = twt)

    model = WebSAMAdapter(encoder, decoder)
    criterion = BCEIoULoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    # Train the model
    train_model(train_dataloader, model, criterion, optimizer, num_epochs=20)