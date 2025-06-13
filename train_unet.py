import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from sklearn.model_selection import train_test_split # No longer needed for pre-split folders
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

## 1. Enhanced U-Net Model Architecture
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Downsampling path
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Upsampling path
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)
        x5 = self.pool(x4)
        
        # Bottleneck
        x5 = self.bottleneck(x5)
        x5 = self.dropout(x5)
        
        # Decoder with skip connections
        u4 = self.up4(x5)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.up_conv4(u4)
        
        u3 = self.up3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up_conv3(u3)
        
        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up_conv2(u2)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up_conv1(u1)
        
        # Output
        output = self.out_conv(u1)
        return torch.sigmoid(output)

## 2. Custom Dataset Class
class CoronaryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths 
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure image and mask are not None (e.g., if file was not found)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        # Ensure masks are binary (0 or 1) if not already, to prevent issues with BCE loss
        mask = (mask > 0).astype(np.float32) # Convert to 0.0 or 1.0

        if self.augment:
            image, mask = self.augment_data(image, mask)
        
        if self.transform:
            # Apply transform to both image and mask
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask
    
    def augment_data(self, image, mask):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            
        # Random rotation (0, 90, 180, 270 degrees)
        rot = np.random.choice([0, 1, 2, 3])
        if rot > 0:
            image = np.rot90(image, rot)
            mask = np.rot90(mask, rot)
            
        return image.copy(), mask.copy() # .copy() to ensure contiguous array after rotation if needed

## 3. Loss Functions (Dice + BCE)
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary Cross Entropy
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Dice Loss
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice_coeff
        
        return BCE + dice_loss

## 4. Training Function
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs=50):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        # Training loop with progress bar
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Validation)'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                
                # Calculate Dice score
                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice_score += (2. * intersection) / (union + 1e-7)
        
        # Calculate metrics
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_dice = dice_score / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice.cpu().numpy())
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print('Model saved!')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_dice_scores, label='Val Dice Score')
    plt.title('Validation Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# --- FUNCTION FOR TEST SET EVALUATION ---
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    dice_score = 0.0
    print("\nEvaluating on Test Set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            test_loss += criterion(outputs, masks).item()
            
            preds = (outputs > 0.5).float()
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice_score += (2. * intersection) / (union + 1e-7)
            
    avg_test_loss = test_loss / len(test_loader)
    avg_test_dice = dice_score / len(test_loader)
    
    print(f'Test Loss: {avg_test_loss:.4f} | Test Dice: {avg_test_dice:.4f}')
    return avg_test_loss, avg_test_dice

# --- HELPER FUNCTION FOR PATHS ---
def get_image_mask_paths(base_dir):
    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_files = sorted(os.listdir(image_dir))
    all_pairs = []
    for img_file in image_files:
        if img_file.endswith('.png'): # Assuming PNG, adjust if other formats
            base_name = os.path.splitext(img_file)[0] # e.g., '1' from '1.png'
            mask_file = f"{base_name}_mask.png" # e.g., '1_mask.png'
            
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))
            else:
                # Warning if a pair is incomplete. This might happen if data isn't perfectly consistent.
                print(f"Warning: Missing corresponding mask or image for {img_file} in {base_dir}")
    return [pair[0] for pair in all_pairs], [pair[1] for pair in all_pairs] # Return separate lists of paths

def main():
    # Hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 200 # Increased epochs as requested
    IMG_SIZE = 256
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Base directory for your dataset splits
    base_data_dir = 'data'

    # --- Load paths for each predefined split ---
    print("Loading training data...")
    train_image_paths, train_mask_paths = get_image_mask_paths(os.path.join(base_data_dir, 'train'))
    
    print("Loading validation data...")
    val_image_paths, val_mask_paths = get_image_mask_paths(os.path.join(base_data_dir, 'vals')) # Note: 'vals' folder
    
    print("Loading test data...")
    test_image_paths, test_mask_paths = get_image_mask_paths(os.path.join(base_data_dir, 'test'))
    
    print(f"Train samples: {len(train_image_paths)}")
    print(f"Validation samples: {len(val_image_paths)}")
    print(f"Test samples: {len(test_image_paths)}")

    if not train_image_paths or not val_image_paths or not test_image_paths:
        print("Error: One or more datasets are empty. Please check your folder structure and file naming.")
        return

    # Create datasets
    train_dataset = CoronaryDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        transform=transform,
        augment=True
    )
    
    val_dataset = CoronaryDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        transform=transform,
        augment=False
    )

    test_dataset = CoronaryDataset(
        image_paths=test_image_paths,
        mask_paths=test_mask_paths,
        transform=transform,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle validation data
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle test data
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS
    )

    # --- Evaluate on the Test Set after training ---
    # Load the best model weights if you want to test the best performing model
    try:
        model.load_state_dict(torch.load('best_unet_model.pth'))
        test_loss, test_dice = test_model(model, test_loader, criterion, device)
        print(f"Final Test Metrics (from best saved model) - Loss: {test_loss:.4f}, Dice: {test_dice:.4f}")
    except FileNotFoundError:
        print("best_unet_model.pth not found. Skipping final test evaluation.")
        print("Please ensure the model was saved successfully during training.")


if __name__ == '__main__':
    main()