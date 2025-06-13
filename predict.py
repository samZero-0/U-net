import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
from tqdm import tqdm # For progress bar during batch evaluation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # Still useful for optional single image visualization

# --- ADD THESE IMPORTS ---
from torch.utils.data import Dataset, DataLoader
# --- END ADDITIONS ---

# Import necessary components from train_unet.py
# Make sure train_unet.py is in the same directory or accessible in your Python path
from train_unet import UNet, CoronaryDataset, get_image_mask_paths 

# --- METRIC CALCULATION FUNCTIONS ---

def calculate_segmentation_metrics(gt_mask, predicted_mask, smooth=1e-7):
    """
    Calculates Dice Score, IoU, Accuracy, Precision, Recall, and Specificity for a single binary mask.

    Args:
        gt_mask (numpy.ndarray): Ground truth binary mask (0s and 1s).
        predicted_mask (numpy.ndarray): Predicted binary mask (0s and 1s).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    if gt_mask is None or predicted_mask is None:
        return {"Dice Score": np.nan, "IoU": np.nan, "Accuracy": np.nan, 
                "Precision": np.nan, "Recall (Sensitivity)": np.nan, "Specificity": np.nan}

    # Flatten the masks for metric calculation
    gt_flat = gt_mask.flatten()
    pred_flat = predicted_mask.flatten()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    # Using confusion matrix for robustness
    # Ensure masks are truly binary (0 or 1)
    gt_flat_int = gt_flat.astype(int)
    pred_flat_int = pred_flat.astype(int)
    
    # labels=[0, 1] ensures the order of TN, FP, FN, TP is consistent
    cm = confusion_matrix(gt_flat_int, pred_flat_int, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Dice Score
    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)

    # IoU (Jaccard Index)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    # Accuracy
    total_pixels = tp + tn + fp + fn
    accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0

    # Precision
    precision = (tp + smooth) / (tp + fp + smooth) 

    # Recall (Sensitivity)
    recall = (tp + smooth) / (tp + fn + smooth) 

    # Specificity
    specificity = (tn + smooth) / (tn + fp + smooth)

    metrics = {
        "Dice Score": dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity
    }
    return metrics

# --- FUNCTION TO EVALUATE ALL IMAGES IN A FOLDER (NEW) ---
def evaluate_folder(model, data_loader, criterion, device):
    """
    Evaluates the model on all images provided by the data_loader and calculates average metrics.

    Args:
        model (nn.Module): The trained U-Net model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate (e.g., test_loader).
        criterion (nn.Module): The loss function (DiceBCELoss).
        device (torch.device): The device (cpu or cuda) to run inference on.

    Returns:
        dict: A dictionary containing the average calculated metrics.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    
    # Initialize sums for TP, FP, FN, TN to calculate overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    print(f"\nEvaluating on {len(data_loader.dataset)} images...")
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            total_loss += criterion(outputs, masks).item() # Accumulate loss

            preds = (outputs > 0.5).float() # Threshold predictions

            # Calculate and accumulate TP, FP, FN, TN for each item in the batch
            for i in range(preds.shape[0]): # Iterate through batch
                gt_flat = masks[i].flatten().cpu().numpy().astype(int)
                pred_flat = preds[i].flatten().cpu().numpy().astype(int)
                
                cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
            
    avg_loss = total_loss / len(data_loader)

    # Calculate overall metrics from accumulated TP, FP, FN, TN
    overall_dice = (2. * total_tp + 1e-7) / (2. * total_tp + total_fp + total_fn + 1e-7)
    overall_iou = (total_tp + 1e-7) / (total_tp + total_fp + total_fn + 1e-7)
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    overall_precision = (total_tp + 1e-7) / (total_tp + total_fp + 1e-7)
    overall_recall = (total_tp + 1e-7) / (total_tp + total_fn + 1e-7)
    overall_specificity = (total_tn + 1e-7) / (total_tn + total_fp + 1e-7)


    avg_metrics = {
        "Average Loss": avg_loss,
        "Overall Dice Score": overall_dice,
        "Overall IoU": overall_iou,
        "Overall Accuracy": overall_accuracy,
        "Overall Precision": overall_precision,
        "Overall Recall (Sensitivity)": overall_recall,
        "Overall Specificity": overall_specificity
    }
    return avg_metrics

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    MODEL_PATH = 'best_unet_model.pth' # Path to your saved model weights
    IMG_SIZE = 256 # Must match the IMG_SIZE used during training
    BATCH_SIZE = 4 # Use the same batch size as training or adjust for test
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Load Model ---
    model = UNet(in_channels=1, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it exists.")
        exit() # Exit if model cannot be loaded

    # --- Define Transformations ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # --- Setup Test DataLoader ---
    base_data_dir = 'data'
    test_image_dir = os.path.join(base_data_dir, 'test', 'images') # Corrected to point directly to images folder
    # get_image_mask_paths function expects the base directory of the split, which includes the 'images' and 'masks' subfolders
    # So, I've adjusted the call to get_image_mask_paths below to reflect this.

    print("Loading test data paths...")
    # The get_image_mask_paths function expects the parent directory of 'images' and 'masks'
    # So, the path should be 'data/test' not 'data/test/images'
    test_image_paths, test_mask_paths = get_image_mask_paths(os.path.join(base_data_dir, 'test')) 

    if not test_image_paths:
        print(f"No image-mask pairs found in {os.path.join(base_data_dir, 'test')}. Cannot perform evaluation.")
        exit()

    test_dataset = CoronaryDataset(
        image_paths=test_image_paths,
        mask_paths=test_mask_paths,
        transform=transform,
        augment=False # No augmentation for evaluation
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Do not shuffle for consistent evaluation order
        num_workers=4 if device.type == 'cuda' else 0, # Use workers if on GPU
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- Define Loss Function ---
    # The DiceBCELoss class is already defined in train_unet.py.
    # We should import it directly from there for consistency.
    from train_unet import DiceBCELoss # Add this import
    criterion = DiceBCELoss() # Instantiate the loss function

    # --- Evaluate the model on the entire test folder ---
    overall_metrics = evaluate_folder(model, test_loader, criterion, device)
    
    print("\n--- Average Segmentation Metrics on Test Set ---")
    for metric_name, value in overall_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Optional: If you still want to visualize a single image after seeing averages
    # You can pick one from the test_image_paths list
    # For example, to visualize the 5th image in your test set:
    # if len(test_image_paths) > 4:
    #     single_image_path_to_visualize = test_image_paths[4]
    #     print(f"\nVisualizing a single test image: {os.path.basename(single_image_path_to_visualize)}")
    #     # Original predict_single_image function is no longer needed if you're only doing folder evaluation
    #     # If you still want it, you'd need to re-add its definition or import it as well.
    #     # For now, it's removed for simplicity if the primary goal is folder evaluation.
    # else:
    #     print("\nNot enough images in test set to visualize a specific index (e.g., 5th image).")