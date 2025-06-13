import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

# Import the UNet model definition from your train-unet.py file
# Make sure train-unet.py is in the same directory or accessible in your Python path
from train_unet import UNet # Assuming your UNet class is in train_unet.py

# --- METRIC CALCULATION FUNCTIONS ---

def calculate_segmentation_metrics(gt_mask, predicted_mask, smooth=1e-7):
    """
    Calculates Dice Score, IoU, Accuracy, Precision, and Recall for a single binary mask.

    Args:
        gt_mask (numpy.ndarray): Ground truth binary mask (0s and 1s).
        predicted_mask (numpy.ndarray): Predicted binary mask (0s and 1s).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    if gt_mask is None or predicted_mask is None:
        return {"Dice Score": np.nan, "IoU": np.nan, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "Specificity": np.nan}

    # Flatten the masks for metric calculation
    gt_flat = gt_mask.flatten()
    pred_flat = predicted_mask.flatten()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    # Using confusion matrix for robustness
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()

    # Dice Score
    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)

    # IoU (Jaccard Index)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = (tp + smooth) / (tp + fp + smooth) # Add smooth to denominator to prevent division by zero

    # Recall (Sensitivity)
    recall = (tp + smooth) / (tp + fn + smooth) # Add smooth to denominator

    # Specificity
    specificity = (tn + smooth) / (tn + fp + smooth) # Add smooth to denominator

    metrics = {
        "Dice Score": dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity
    }
    return metrics

# --- EXISTING predict_single_image FUNCTION (NO CHANGE) ---
def predict_single_image(model_path, image_path, img_size, device):
    """
    Loads a trained U-Net model, performs inference on a single image,
    and returns the predicted mask.
    """
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    original_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    input_img_tensor = transform(original_img_cv).unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        output = model(input_img_tensor)

    predicted_mask_tensor = (output > 0.5).float() # Threshold at 0.5
    predicted_mask = predicted_mask_tensor.squeeze(0).squeeze(0).cpu().numpy() # Remove batch & channel dims, to numpy

    gt_mask = None
    image_dir = os.path.dirname(image_path)
    base_split_dir = os.path.dirname(image_dir) # e.g., 'data/test'
    mask_folder = os.path.join(base_split_dir, 'masks')
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    gt_mask_filename = f"{base_filename}_mask.png"
    gt_mask_path = os.path.join(mask_folder, gt_mask_filename)

    if os.path.exists(gt_mask_path):
        gt_mask_cv = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask_cv is not None:
            gt_mask = (cv2.resize(gt_mask_cv, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
            
    return original_img_cv, gt_mask, predicted_mask

# --- EXISTING visualize_prediction FUNCTION (NO CHANGE) ---
def visualize_prediction(original_img, gt_mask, predicted_mask, img_size):
    """
    Displays the original image, ground truth mask, and predicted mask.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    if gt_mask is not None:
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth Mask')
    else:
        plt.text(0.5, 0.5, 'Ground Truth Mask Not Found', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='red')
        plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    MODEL_PATH = 'best_unet_model.pth' # Path to your saved model weights
    IMG_SIZE = 256 # Must match the IMG_SIZE used during training
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    test_image_dir = 'data/test/images'
    INPUT_IMAGE_PATH = None
    
    if os.path.exists(test_image_dir):
        test_images = sorted(os.listdir(test_image_dir))
        if test_images:
            # Pick a specific image. You can change the index here, e.g., test_images[0], test_images[10], etc.
            # I've used index 3 as in your example, but feel free to pick another.
            test_image_filename = test_images[7]
            INPUT_IMAGE_PATH = os.path.join(test_image_dir, test_image_filename)
            print(f"Using test image: {INPUT_IMAGE_PATH}")
        else:
            print(f"No images found in {test_image_dir}. Please ensure your test data is there.")
    else:
        print(f"Test image directory not found: {test_image_dir}. Please check your path.")

    if INPUT_IMAGE_PATH:
        try:
            original_img, gt_mask, predicted_mask = predict_single_image(MODEL_PATH, INPUT_IMAGE_PATH, IMG_SIZE, device)
            
            # --- Calculate and print metrics ---
            if gt_mask is not None:
                metrics = calculate_segmentation_metrics(gt_mask, predicted_mask)
                print("\n--- Segmentation Metrics ---")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
            else:
                print("\nCannot calculate metrics: Ground truth mask not found for the selected image.")

            visualize_prediction(original_img, gt_mask, predicted_mask, IMG_SIZE)

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    else:
        print("Could not find an image to predict. Exiting.")