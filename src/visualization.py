from tqdm import tqdm
import os
import torch
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil
from PIL import Image
from albumentations.pytorch import ToTensorV2 
import albumentations as A
from sklearn.metrics import confusion_matrix
import sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.training import calculate_segmentation_metrics,compute_metrics_from_stats

from src.transform import get_val_test_transform

def generate_merged_images(model_name,model,loader,prediction_folder,dataset_name):
    merged_image_folder = os.path.join(prediction_folder,"merged_predictions",dataset_name)
    os.makedirs(merged_image_folder, exist_ok=True) 
    model.eval()
    image_counter = 0
    
    for batch in loader:

        images = batch[0].to(device)
        masks = batch[1].to(device)


        with torch.no_grad():
            print("model_name",model_name)
            if model_name.lower() in ["matsegnet"]:
                
                masks=masks.squeeze(1)
                outputs = model(images)[0]
                predicted_masks = torch.sigmoid(outputs)
                predicted_masks = (predicted_masks > 0.5).float()
                predicted_masks=predicted_masks.squeeze(1)
            elif model_name.lower()=="segformer":
                outputs = model(pixel_values=images)
                logits = outputs.logits

                upsampled_logits = torch.nn.functional.interpolate(
                    logits, 
                    size=images.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )

                predicted_masks = torch.argmax(upsampled_logits, dim=1)

            elif model_name.lower() in ["unet", "fpn"]:
                outputs = model(images)
                predicted_masks = torch.sigmoid(outputs)
                predicted_masks = (predicted_masks > 0.5).float()
                predicted_masks = predicted_masks.squeeze(1)

        for i in range(images.shape[0]):
            img_to_show = images[i]
            truth_mask_to_show = masks[i]
            pred_mask_to_show = predicted_masks[i]

            display_list = [img_to_show, truth_mask_to_show, pred_mask_to_show, img_to_show]

  
            display_with_colored_masks(display_list, image_counter, merged_image_folder)

    
            image_counter += 1

    print(f"Finished. All predictions saved to the '{merged_image_folder}' directory.")
    

def analyze_and_visualize_predictions(loader, model, model_name, device, num_to_show=1):

    model.eval()
    iou_scores = []
    images_shown = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Analyzing Predictions", total=num_to_show)
        
   
        for batch in progress_bar:
            if images_shown >= num_to_show:
                break

            if model_name.lower() in ["matsegnet"]:
                images_batch, masks_batch, _ = batch
            else:
                images_batch, masks_batch = batch
            
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

       
            outputs_batch = model(images_batch)
 
            if model_name.lower() == "segformer":
                logits = outputs_batch.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=images_batch.shape[-2:], mode='bilinear', align_corners=False
                )
                pred_masks_batch = torch.argmax(upsampled_logits, dim=1)
                masks_batch = masks_batch.squeeze(1).long() 
            
            elif model_name.lower() in ["unet", "matsegnet","fpn"]:
                if model_name.lower() in  ["matsegnet"]:
                    outputs_batch = outputs_batch[0]
                
                pred_probs = torch.sigmoid(outputs_batch)
                pred_masks_batch = (pred_probs > 0.5).float()
                pred_masks_batch = pred_masks_batch.squeeze(1)
                masks_batch = masks_batch.squeeze(1).float() 

    
            for i in range(images_batch.size(0)):
                if images_shown >= num_to_show:
                    break
                
                print(f"\n--- Analyzing Sample {images_shown + 1} ---")
                
                image_tensor = images_batch[i]
                true_mask_tensor = masks_batch[i]
                pred_mask_tensor = pred_masks_batch[i]
                
 
                true_mask_np = true_mask_tensor.cpu().numpy()
                pred_mask_np = pred_mask_tensor.cpu().numpy()
                y_true = true_mask_np.flatten()
                y_predict = pred_mask_np.flatten()
                
                labels = np.unique(np.concatenate((y_true, y_predict)))
                conf_matrix = confusion_matrix(y_true, y_predict, labels=labels)
 
                iou_result = compute_iou_carbides(conf_matrix) 
                iou_scores.append(iou_result)
                print(f"Mean IoU: {iou_result:.4f}")

   
                error_map_tensor = create_error_visualization_mask(pred_mask_tensor, true_mask_tensor)
                
                image_for_viz = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                colorized_true_mask = colorize_mask(true_mask_tensor)
                colorized_pred_mask = colorize_mask(pred_mask_tensor)
                colorized_error_map = colorize_mask(error_map_tensor)

                overlay_truth = cv2.addWeighted(image_for_viz, 0.6, colorized_true_mask, 0.4, 0)
                overlay_pred = cv2.addWeighted(image_for_viz, 0.6, colorized_pred_mask, 0.4, 0)
                overlay_error = cv2.addWeighted(image_for_viz, 0.6, colorized_error_map, 0.4, 0)

                
                images_shown += 1
                progress_bar.update(1)
                    
    return iou_scores
	
    

    
def get_iou_list(dataset_loader, model, model_name,num=None, device=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    
    model.to(device)
    model.eval()
    iou_list = []

    with torch.no_grad():
        progress_bar = tqdm(dataset_loader, desc=f"Calculating IoU for {model_name}")
        for i,batch  in enumerate(progress_bar):
            if num is not None and i >= num:
                break
                    
            if model_name.lower() in ["matsegnet"]:
                images_batch,masks_batch,_ = batch
            else:
                images_batch, masks_batch = batch 

            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)  
            outputs_batch = model(images_batch) 
                
                
            if model_name.lower() == "segformer":
                logits = outputs_batch.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, 
                    size=images_batch.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                pred_masks_batch = torch.argmax(upsampled_logits, dim=1)
                masks_batch = masks_batch.squeeze(1).long()
            
            elif model_name.lower() in ["unet", "matsegnet","fpn"]:
                if model_name.lower()  in ["matsegnet"]:
                    outputs_batch = outputs_batch[0] 
                
                pred_probs = torch.sigmoid(outputs_batch)
                pred_masks_batch = (pred_probs > 0.5).float()
                pred_masks_batch = pred_masks_batch.squeeze(1) 
                masks_batch = masks_batch.squeeze(1).float()
            
            for j in range(images_batch.size(0)):
                true_mask_tensor = masks_batch[j]
                pred_mask_tensor = pred_masks_batch[j]


                mask_np = true_mask_tensor.cpu().numpy()
                pred_np = pred_mask_tensor.cpu().numpy()

                y_true = mask_np.flatten()
                y_predict = pred_np.flatten()


                labels = np.unique(np.concatenate((y_true, y_predict)))
                conf_matrix = confusion_matrix(y_true, y_predict, labels=labels)
                

                iou_result = compute_iou_carbides(conf_matrix)

                iou_list.append(float(iou_result))
                        
    return iou_list

    
def plot_training_history(history_dict,prediction_file):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    

    acc = history_dict['train_f1_score']
    val_acc = history_dict['val_f1_score']
    loss = history_dict['train_loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)


    plt.rcParams["font.family"] = "Times New Roman"
    csfont = {'fontname':'Times New Roman'}
    ax1.plot(epochs, acc, 'b^-', label='Training F1 Score', linewidth=2)
    ax1.plot(epochs, val_acc, 'ro-', label='Validation F1 Score', linewidth=2)
    ax1.set_title('Training and Validation F1 Score', fontsize=20, **csfont)
    ax1.set_xlabel('Epochs', fontsize=16, **csfont)
    ax1.set_ylabel('F1 Score', fontsize=16, **csfont)
    ax1.legend(fontsize=14)
   # ax1.grid(False, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]

    # --- Plot 2: Loss ---
    ax2.plot(epochs, loss, 'b^-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'ro-', label='Validation Loss', linewidth=2)
    ax2.set_title('Training and Validation Loss', fontsize=20, **csfont)
    ax2.set_xlabel('Epochs', fontsize=16, **csfont)
    ax2.set_ylabel('Loss', fontsize=16, **csfont)
    ax2.legend(fontsize=14)
   # ax2.grid(False, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    [x.set_linewidth(1.5) for x in ax2.spines.values()]

    plt.tight_layout()
    plt.savefig(prediction_file, dpi=300)



def compute_iou_carbides(confusion_matrix):
    """Computes the Intersection over Union (IoU) for carbides from a NumPy confusion matrix."""
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    return iou[1]
    
    
    

def save_iou_results_to_file(filename, train_scores, val_scores, test_scores):
    """Saves the IoU scores for each dataset to a text file."""
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as f:
        f.write("--- Training Set IoU Scores ---\n")
        f.write(','.join([f"{score:.4f}" for score in train_scores]))
        f.write("\n\n")
        
        f.write("--- Validation Set IoU Scores ---\n")
        f.write(','.join([f"{score:.4f}" for score in val_scores]))
        f.write("\n\n")

        f.write("--- Test Set IoU Scores ---\n")
        f.write(','.join([f"{score:.4f}" for score in test_scores]))
        f.write("\n")
    print(f"IoU results saved to {filename}")
    
    

def plot_iou_scatter(train_scores, val_scores, test_scores, train_loader, val_loader, test_loader,prediction_file):
 
    train_image_number = len(train_loader.dataset)
    validation_image_number = len(val_loader.dataset)
    test_image_number = len(test_loader.dataset)


    train_x = np.arange(train_image_number)
    val_x = train_image_number + np.arange(validation_image_number)
    test_x = train_image_number + validation_image_number + np.arange(test_image_number)

    fig, ax = plt.subplots(figsize=(16, 9))
    

    marker_size = 60
    edge_width=2
    ax.scatter(train_x, train_scores, label='Training Data', alpha=0.7, marker='^',s=marker_size,facecolors='none',edgecolors='blue',linewidths=edge_width) 
    ax.scatter(val_x, val_scores, label='Validation Data', alpha=0.7, marker='>',s=marker_size,facecolors='none',edgecolors='orange',linewidths=edge_width)
    ax.scatter(test_x, test_scores, label='Testing Data', alpha=0.7, marker='v',s=marker_size,facecolors='none',edgecolors='green',linewidths=edge_width)
    

    plt.rcParams["font.family"] = "Times New Roman"
    csfont = {'fontname':'Times New Roman'}
    ax.set_xlabel('Image Index', fontsize=20, **csfont)
    ax.set_ylabel('Mean IoU Score', fontsize=20, **csfont)
    ax.set_title('IoU Score for Each Image in All Datasets', fontsize=24, **csfont)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8, direction='in')
    [x.set_linewidth(1.5) for x in ax.spines.values()]
   # plt.grid(False, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(os.path.join(prediction_file), dpi=300)




    
    

def save_training_history(history_dict, prediction_folder):
    """Saves the training history dictionary to a text file."""
    filename=os.path.join(prediction_folder,'LossAndAccuracy.txt')
    with open(filename, "w") as f:
        for key, value_list in history_dict.items():
            # Write the metric name as a header
            f.write(f"--- {key} ---\n")
            # Write the values as a comma-separated string
            f.write(','.join([f"{value:.4f}" for value in value_list]))
            f.write("\n\n")
    print(f"Training history saved to {filename}")



def display_with_colored_masks(display_list, num,prediction_folder):
    """
    show masks, original image, and predictions
    """
    plt.figure(figsize=(20, 5))
    title = ['(a) Image', '(b) Ground Truth Mask', '(c) NN Output', '(d) Overlay']

    processed_list = []

    raw_true_mask = None
    raw_pred_mask = None
    
    
    for i, tensor in enumerate(display_list):
        tensor = tensor.cpu().detach().squeeze(0)
        if i == 0:  
            np_image = tensor.permute(1, 2, 0).numpy()
            np_image = np.clip(np_image, 0, 1)
            processed_list.append(np_image)
        elif i == 1:  
            raw_true_mask = tensor
            colored_np_array = colorize_mask(tensor)
            processed_list.append(colored_np_array)
        elif i == 2:  
            raw_pred_mask = tensor
   #         colored_np_array = colorize_mask(tensor)
            mask_to_overlay_np = create_error_visualization_mask(raw_pred_mask, raw_true_mask) #
            mask_to_overlay_np = colorize_mask(mask_to_overlay_np)#
            processed_list.append(mask_to_overlay_np) #
          #  processed_list.append(colored_np_array)
        else: 
            original_image_np = processed_list[0]
    
            mask_to_overlay_np = create_error_visualization_mask(raw_pred_mask, raw_true_mask)
            mask_to_overlay_np=colorize_mask(mask_to_overlay_np)
            
            img_uint8 = (original_image_np * 255).astype(np.uint8)
            
            if isinstance(mask_to_overlay_np, torch.Tensor):
                mask_to_overlay_np = mask_to_overlay_np.cpu().numpy()

            if mask_to_overlay_np.dtype != 'uint8':
                mask_to_overlay_np = mask_to_overlay_np.astype(np.uint8)
            overlay = cv2.addWeighted(img_uint8, 0.6, mask_to_overlay_np, 0.4, 0)
            processed_list.append(overlay)

    
    for i in range(len(processed_list)):
        plt.subplot(1, len(processed_list), i + 1)
        plt.title(title[i], fontsize=20)
        plt.imshow(processed_list[i])
        plt.axis('off')
    if not os.path.exists(os.path.join(prediction_folder)):
        os.mkdir(os.path.join(prediction_folder))
    plt.savefig(os.path.join(prediction_folder,f'segmentation_merge_{num}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()    

def colorize_mask(mask):
    """
    Converts a single-channel integer mask to a 3-channel RGB color mask
    using fast, vectorized NumPy operations.

    Args:
        mask (np.ndarray or torch.Tensor): The input mask of shape [H, W] with integer class labels.

    Returns:
        np.ndarray: A colorized mask of shape [H, W, 3] with dtype=uint8.
    """
    # 1. Ensure the input is a NumPy array
    if torch.is_tensor(mask):
        # Move tensor to CPU and convert to NumPy if it's a PyTorch tensor
        mask = mask.cpu().numpy()

    # 2. Define the color map
    # This maps the integer class label (the key) to an RGB color (the value)
    color_map = {
        0: [216, 191, 216],  # Background
        1: [255, 255, 0],   
        3: [255, 0, 0],      # Class 3
        25: [0, 255, 0]      # Class 25
    }

    # 3. Create the output RGB image
    # We start with a blank (black) canvas of the correct size and data type
    output_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # 4. Apply the colors in a vectorized way
    # This is the fast part. Instead of looping, we use boolean indexing.
    for class_id, color in color_map.items():
        # Find all locations (pixels) where the mask has the current class_id
        locations = (mask == class_id)
        # For all these locations, assign the corresponding color at once
        output_mask[locations] = color

    return output_mask
    
    


def create_error_visualization_mask(pred_mask, true_mask):  #create mask 2
    """
    Creates a visualization mask to highlight False Positives and False Negatives.
    This function expects single, non-batched masks that have already been
    processed by argmax (i.e., they contain class indices).

    Args:
        pred_mask (torch.Tensor): The predicted mask, shape [H, W], with integer class labels (0, 1, ...).
        true_mask (torch.Tensor): The ground truth mask, shape [H, W], with integer class labels.

    Returns:
        torch.Tensor: A new mask of shape [H, W] where False Positives and 
                      False Negatives are marked with special values.
    """
  
    pred_mask = pred_mask.clone().detach().long()
    true_mask = true_mask.clone().detach().long()  

    # --- 1. Calculate the difference to identify error types ---
    # We cast to float to allow for negative results (e.g., 0 - 1 = -1).
    # A correct prediction results in a difference of 0.
    difference = pred_mask.float() - true_mask.float()

    # --- 2. Identify and count errors ---
    # False Negative (FN): Prediction is 0, Ground Truth is 1. Difference is -1.
    fn_pixels = torch.sum(difference < 0).item()
    print(f"Summation of False Negatives (FN): {fn_pixels}")

    # False Positive (FP): Prediction is 1, Ground Truth is 0. Difference is +1.
    fp_pixels = torch.sum(difference > 0).item()
    print(f"Summation of False Positives (FP): {fp_pixels}")

    tp_pixels = torch.sum((difference == 0) & (pred_mask == 1)).item()
    print(f"Summation of True Positives (TP): {tp_pixels}")

# True Negative (TN): Prediction=0, GT=0 → difference = 0 且 pred=0
    tn_pixels = torch.sum((difference == 0) & (pred_mask == 0)).item()
    print(f"Summation of True Negatives (TN): {tn_pixels}")

    # --- 3. Create the error mask using torch.where ---
    # torch.where(condition, value_if_true, value_if_false)
    
    # Define the special values for errors, making them clear constants
    FN_MARKER = 25  # Value to assign to False Negative pixels
    FP_MARKER = 3   # Value to assign to False Positive pixels

    # Start with a copy of the original prediction. This will be our canvas.
    # Correct predictions (0 -> 0, 1 -> 1) will be preserved unless overwritten.
    error_mask = pred_mask.clone()

    # First, mark the False Negatives on our canvas
    error_mask = torch.where(
        difference < 0,            # Condition: Is it a False Negative?
        torch.tensor(FN_MARKER),   # If yes, set pixel to 25
        error_mask                 # If no, keep the current value
    )

    # Next, on the *result* of the previous step, mark the False Positives
    error_mask = torch.where(
        difference > 0,            # Condition: Is it a False Positive?
        torch.tensor(FP_MARKER),   # If yes, set pixel to 3
        error_mask                 # If no, keep the value (which might be a correct prediction or the FN_MARKER)
    )

    return error_mask
    
    
    





def save_error_map_visualizations(loader, model, model_name,device, output_path, name_prefix, resize_dim=(512, 512)):
    """
    Generates colorized error map visualizations for an entire dataset and saves them to disk.
    This is a memory-efficient replacement for 'save_predictions_superposition'.
    """
    print(f"--- Preparing to save error map visualizations to: {output_path} ---")
    
    # 1. Safely create the output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # Deletes the folder if it exists
    os.makedirs(output_path)      # Creates a fresh folder

    model.eval()  # Set the model to evaluation mode
    sample_index = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Saving '{name_prefix}' error maps")
        
        # This time we need both the image and the ground truth mask
        for batch in progress_bar:
            images_batch = batch[0].to(device)
            masks_batch = batch[1].to(device)
            
            # Get model predictions
            outputs_batch = model(images_batch)
                          
           
            
            if model_name.lower() in ["unet","fpn"]:
 
                pred_masks_batch = torch.sigmoid(outputs_batch)
                pred_masks_batch = (pred_masks_batch > 0.5).float()
                pred_masks_batch = pred_masks_batch.squeeze(1)
                
            if model_name.lower() =="segformer":
                outputs_batch = outputs_batch.logits
                outputs_batch = torch.nn.functional.interpolate(outputs_batch, size=images_batch.shape[-2:], mode='bilinear', align_corners=False)
                pred_masks_batch = torch.argmax(outputs_batch, dim=1)
                
            if model_name.lower()  in ["matsegnet"]:
                outputs_batch = outputs_batch[0]
                outputs_batch = torch.sigmoid(outputs_batch)
                pred_masks_batch = (outputs_batch > 0.5).float()
                pred_masks_batch=pred_masks_batch.squeeze(1)
                print(pred_masks_batch.shape)
             

            # Process and save each image in the batch
            for i in range(images_batch.size(0)):
                true_mask_tensor = masks_batch[i]
                pred_mask_tensor = pred_masks_batch[i]
                
                # Create the error map (e.g., with values 0, 1, 3, 25)
                error_map_tensor = create_error_visualization_mask(pred_mask_tensor, true_mask_tensor)

                # Colorize the error map (returns a NumPy array in RGB format)
                
                error_map_tensor=error_map_tensor.squeeze(0)
                colorized_error_map_rgb = colorize_mask(error_map_tensor)
                
                # Resize the colorized error map
                resized_error_map_rgb = cv2.resize(colorized_error_map_rgb, resize_dim, interpolation=cv2.INTER_AREA)
                
                # Convert from RGB to BGR for saving with OpenCV
                resized_error_map_bgr = cv2.cvtColor(resized_error_map_rgb, cv2.COLOR_RGB2BGR)

                # Construct the full, safe file path
                save_path = os.path.join(output_path, f"{name_prefix}_{sample_index}.jpg")
                
                # Save the final image to disk
                cv2.imwrite(save_path, resized_error_map_bgr)
                
                sample_index += 1
                
    print(f"--- Successfully saved {sample_index} error map images to {output_path} ---")
    
    
    
    


def predict_and_save_folder(model, model_name, device, input_folder, output_folder, input_resize_dim, output_resize_dim=(512, 512)):
    """
    Loads all images from a folder, runs model predictions, and saves the 
    colorized, resized output masks.
    """
    print(f"\n--- Processing folder for size analysis: {input_folder} ---")
    
    # 1. Safely create the output directory
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 2. Define a transformation pipeline just for this inference task
    #    This should match the validation/test transform your model expects.
   
    model.eval()
    
    # 3. Iterate through all files in the input folder
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg'))]
    inference_transform = get_val_test_transform(img_height=input_resize_dim[1], img_width=input_resize_dim[0])
    with torch.no_grad():
        for filename in tqdm(filenames, desc=f"Predicting images in {os.path.basename(input_folder)}"):
            # Construct the full image path
            img_path = os.path.join(input_folder, filename)
            
            # Load the image using Pillow and convert to RGB
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Could not read image {filename}, skipping. Error: {e}")
                continue

            # Apply the transformations to create a tensor

            transformed = inference_transform(image=image)
            image_tensor = transformed['image'].to(device)
            
            # Add a batch dimension (C, H, W) -> (1, C, H, W) and run prediction
            output = model(image_tensor.unsqueeze(0))
            
            if model_name.lower() =="unet":
                
                pred_mask = torch.sigmoid(output)
                pred_mask = (pred_mask > 0.5).float()
                pred_mask = pred_mask.squeeze(0)         
                pred_mask = pred_mask.squeeze(0)   
            if model_name.lower() =="segformer":
                output = output.logits
                output = torch.nn.functional.interpolate(output, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
                pred_mask = torch.argmax(output, dim=1).squeeze(0) 
            if model_name.lower() in ["matsegnet"]:
                output = output[0]
                output = torch.sigmoid(output)
                output = (output > 0.5).float()
                pred_mask=output.squeeze(0)
                pred_mask=pred_mask.squeeze(0)
            if model_name.lower() =="fpn":
                pred_mask = torch.sigmoid(output)
                pred_mask = (pred_mask > 0.5).float()
                pred_mask = pred_mask.squeeze(0)         
                pred_mask = pred_mask.squeeze(0)  

            colorized_pred = colorize_mask(pred_mask) # Use your existing helper
            
            # Resize to the final output dimensions
            resized_output = cv2.resize(colorized_pred, output_resize_dim, interpolation=cv2.INTER_AREA)
            
            # Convert RGB to BGR for saving with OpenCV
            final_image_to_save = cv2.cvtColor(resized_output, cv2.COLOR_RGB2BGR)
            
            # Save the final image
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, final_image_to_save)

    print(f"--- Finished processing. Saved results to {output_folder} ---")


def write_test_results_to_file(results_dict, output_dir, model_name, filename="test_results.txt"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(filepath, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"Test Results for Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Test set accuracy: {results_dict.get('accuracy', 'N/A')}\n")
        f.write(f"Test set recall: {results_dict.get('recall', 'N/A')}\n")
        f.write(f"Test set precision: {results_dict.get('precision', 'N/A')}\n")
        f.write(f"Test set F1 score: {results_dict.get('f1_score', 'N/A')}\n")
        f.write("="*50 + "\n\n")
        
    print(f"[*] Test results successfully written to: {filepath}")


def final_evaluation(loader, model, model_name, device):

    model.eval()
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        
    with torch.no_grad(): 
        progress_bar = tqdm(loader, desc="Final Test Set Evaluation")
        
        for batch in progress_bar:

            if model_name.lower() in ["matsegnet"]:
                images, masks, _ = batch
            else:
                images, masks = batch
            
            images = images.to(device)
            masks = masks.to(device)

            if model_name == "Segformer":
                masks = masks.squeeze(1).long()
                #segformer_masks=masks.long()
                outputs = model(pixel_values=images) #, labels=segformer_masks

                logits = outputs.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
                preds = torch.argmax(upsampled_logits, dim=1)
            elif model_name.lower() in ["unet", "matsegnet","fpn"]:
                outputs = model(images)

                if model_name.lower() in ["matsegnet"]:
                    mask_logits = outputs[0]

                  
                else:
                    mask_logits = outputs
            
                
                pred_probs = torch.sigmoid(mask_logits)
                preds = (pred_probs > 0.5).float()

            batch_stats = calculate_segmentation_metrics(preds, masks)
            epoch_stats['tp'] += batch_stats['tp']
            epoch_stats['fp'] += batch_stats['fp']
            epoch_stats['fn'] += batch_stats['fn']
            epoch_stats['tn'] += batch_stats['tn']

 
    final_metrics = compute_metrics_from_stats(epoch_stats)

    

    results = {
        'f1_score': final_metrics['f1'],
        'accuracy': final_metrics['accuracy'],
        'recall': final_metrics['recall'],
        "precision":final_metrics['precision']
     
    }
    
    print("\n--- Final Evaluation Summary ---")
    for key, value in results.items():
        print(f"  - {key.replace('_', ' ').title()}: {value:.4f}")
        
    return results
    
    
def plot_training_history_custom(history_dict,prediction_folder):
    """
    Plots the training and validation accuracy and loss with custom formatting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Extract data from the history dictionary
    acc = history_dict['train_acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['train_loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # --- Plot 1: Accuracy ---
    csfont = {'fontname': 'Times New Roman'}
    ax1.plot(epochs, acc, 'bo-', label='Training Data', linewidth=2)
    ax1.plot(epochs, val_acc, 'ro-', label='Validation Data', linewidth=2)
    ax1.set_xlabel('Epoch Number', fontsize=24, **csfont)
    ax1.set_ylabel('Accuracy', fontsize=24, **csfont)
    ax1.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10, direction='in')
    [x.set_linewidth(1.5) for x in ax1.spines.values()]

    # --- Plot 2: Loss ---
    ax2.plot(epochs, loss, 'bo-', label='Training Data', linewidth=2)
    ax2.plot(epochs, val_loss, 'ro-', label='Validation Data', linewidth=2)
    ax2.set_xlabel('Epoch Number', fontsize=24, **csfont)
    ax2.set_ylabel('Loss', fontsize=24, **csfont)
    ax2.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10, direction='in')
    [x.set_linewidth(1.5) for x in ax2.spines.values()]

    # --- Common Legend for both plots ---
    # The legend is added to the figure, not the individual axes, for a shared title effect.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               fancybox=True, shadow=True, ncol=2, fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for the figure title
    plt.savefig(os.path.join(prediction_folder,'LossAndAccuracy.png'), dpi=300)




def save_all_predictions(loader, model, model_name,device, output_path, name_prefix, resize_dim=(512, 512)):
    """
    Runs model predictions for an entire dataset, colorizes the masks,
    resizes them, and saves them to a specified folder.
    This is a memory-efficient replacement for the original Keras code.
    """
    print(f"--- Preparing to save predictions to: {output_path} ---")
    
 
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  
    os.makedirs(output_path)  

    model.eval() 
    sample_index = 0
    
    with torch.no_grad(): 
        
        progress_bar = tqdm(loader, desc=f"Saving '{name_prefix}' predictions")


        for batch in progress_bar: 
            images_batch=batch[0]
            images_batch = images_batch.to(device)
            outputs_batch = model(images_batch)
            
            
            if model_name.lower() in ["unet" , "fpn"]:
                pred_masks_batch = torch.sigmoid(outputs_batch)
                pred_masks_batch = (pred_masks_batch > 0.5).float()
                pred_masks_batch = pred_masks_batch.squeeze(1)
       
                
            if model_name.lower() =="segformer":
                outputs_batch = outputs_batch.logits
                outputs_batch = torch.nn.functional.interpolate(outputs_batch, size=images_batch.shape[-2:], mode='bilinear', align_corners=False)
                pred_masks_batch = torch.argmax(outputs_batch, dim=1)
                
            if model_name.lower() in ["matsegnet"]:
                outputs_batch = outputs_batch[0]
                outputs_batch = torch.sigmoid(outputs_batch)
                pred_masks_batch = (outputs_batch > 0.5).float()
                pred_masks_batch=pred_masks_batch.squeeze(1)
             

            # Process and save each image in the current batch
            for i in range(pred_masks_batch.size(0)):
                pred_mask_tensor = pred_masks_batch[i]
                
                # Colorize the mask (this returns a NumPy array in RGB format)
                colorized_pred_rgb = colorize_mask(pred_mask_tensor)
                
                # Resize the colorized mask
                resized_pred_rgb = cv2.resize(colorized_pred_rgb, resize_dim, interpolation=cv2.INTER_AREA)
                
                # Convert from RGB (standard) to BGR for saving with OpenCV
                resized_pred_bgr = cv2.cvtColor(resized_pred_rgb, cv2.COLOR_RGB2BGR)

                # Construct the full, safe file path without changing the working directory
                save_path = os.path.join(output_path, f"{name_prefix}_{sample_index}.jpg")
                
                # Save the final image to disk
                cv2.imwrite(save_path, resized_pred_bgr)
                
                sample_index += 1
                
    print(f"--- Successfully saved {sample_index} prediction images to {output_path} ---")
    
    