import os
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
import torch
import sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.transform import get_train_transform,get_val_test_transform


def get_data_paths(dataset_directory):
    paths = {
        'mask_train': os.path.join(dataset_directory,"training_set","mask"),
        'image_train': os.path.join(dataset_directory,"training_set","original"),
        'edge_train': os.path.join(dataset_directory,"training_set","edge"),
        'mask_validation': os.path.join(dataset_directory,"validation_set","mask"),
        'image_validation': os.path.join(dataset_directory,"validation_set","original"),
        'edge_validation': os.path.join(dataset_directory,"validation_set","edge"),
        'mask_test': os.path.join(dataset_directory,"test_set","mask"),
        'image_test':  os.path.join(dataset_directory,"test_set","original"),
        'edge_test': os.path.join(dataset_directory,"test_set","edge"),
        'bainite_mask_train': os.path.join(dataset_directory,"bainite_set","mask"),
        'bainite_image_train': os.path.join(dataset_directory,"bainite_set","original"),
        'bainite_edge_train': os.path.join(dataset_directory,"bainite_set","edge"),
        'martensite_mask_train': os.path.join(dataset_directory,"martensite_set","mask"),
        'martensite_image_train': os.path.join(dataset_directory,"martensite_set","original"),
        'martensite_edge_train': os.path.join(dataset_directory,"martensite_set","edge")
    }
    return paths



    
def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size=8, num_workers=0, pin_memory=True,
                       test_batch=True):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if test_batch:
        print("Testing the train_loader...")
        parameter_number=(len(next(iter(train_loader))))
        if parameter_number==2:
            images, masks = next(iter(train_loader))
        elif parameter_number==3:
            images, masks,edges = next(iter(train_loader))
            print(f"Edges batch shape: {edges.shape}")
            print(f"Edge dtype: {edges.dtype}")         
        print(f"Images batch shape: {images.shape}") 
        print(f"Masks batch shape: {masks.shape}")   
        print(f"Image dtype: {images.dtype}")       
        print(f"Mask dtype: {masks.dtype}")     

    return train_loader, val_loader, test_loader



def create_datasets(image_list_train, mask_list_train,
                    image_list_validation, mask_list_validation,
                    image_list_test, mask_list_test,
                    img_height, img_width):


    train_transform = get_train_transform(img_height, img_width)
    val_test_transform = get_val_test_transform(img_height, img_width)
    # ==========================================================
    # Dataset creation
    train_dataset = SegmentationDataset(
        image_paths=image_list_train,
        mask_paths=mask_list_train,
        transform=train_transform
    )

    validation_dataset = SegmentationDataset(
        image_paths=image_list_validation,
        mask_paths=mask_list_validation,
        transform=val_test_transform
    )

    test_dataset = SegmentationDataset(
        image_paths=image_list_test,
        mask_paths=mask_list_test,
        transform=val_test_transform
    )
    return train_transform,val_test_transform,train_dataset, validation_dataset, test_dataset
    
    
def create_edge_datasets(image_list_train, mask_list_train,edge_list_train,
                    image_list_validation, mask_list_validation,edge_list_validation,
                    image_list_test, mask_list_test,edge_list_test,
                    img_height, img_width):



    train_transform = get_train_transform(img_height, img_width)
    val_test_transform = get_val_test_transform(img_height, img_width)

    train_dataset = SegmentationEdgeDataset(
        image_paths=image_list_train,
        mask_paths=mask_list_train,
        edge_paths=edge_list_train,
        transform=train_transform
    )

    validation_dataset = SegmentationEdgeDataset(
        image_paths=image_list_validation,
        mask_paths=mask_list_validation,
        edge_paths=edge_list_validation,
        transform=val_test_transform
    )

    test_dataset = SegmentationEdgeDataset(
        image_paths=image_list_test,
        mask_paths=mask_list_test,
        edge_paths=edge_list_test,
        transform=val_test_transform
    )
    return train_transform,val_test_transform,train_dataset, validation_dataset, test_dataset
  
    
    

def get_sorted_image_mask_lists(image_dir, mask_dir):
    # List files
    image_list = os.listdir(image_dir)
    mask_list = os.listdir(mask_dir)

    # Prepend full path
    image_list = [os.path.join(image_dir, f) for f in image_list]
    mask_list = [os.path.join(mask_dir, f) for f in mask_list]

    # Sort lists to keep correspondence
    image_list.sort()
    mask_list.sort()

    return (image_list,mask_list)

def get_sorted_image_mask_edge_lists(image_dir, mask_dir, edge_dir):
    # List files
    image_list = os.listdir(image_dir)
    mask_list = os.listdir(mask_dir)
    edge_list = os.listdir(edge_dir)
    
    # Prepend full path
    image_list = [os.path.join(image_dir, f) for f in image_list]
    mask_list = [os.path.join(mask_dir, f) for f in mask_list]
    edge_list = [os.path.join(edge_dir, f) for f in edge_list]

    # Sort lists to keep correspondence
    image_list.sort()
    mask_list.sort()
    edge_list.sort()
    
    return (image_list,mask_list,edge_list)





class SegmentationDataset(Dataset):
    """
    Custom Dataset for Image Segmentation.
    Reads image and mask paths, loads them, and applies transformations.
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        green_channel = mask_rgb[:, :, 1]
        mask = (green_channel > 127).astype(np.float32)

        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.unsqueeze(0)
        

class SegmentationEdgeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, edge_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.edge_paths = edge_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = np.array(Image.open(self.image_paths[index]).convert("RGB"))
         
        mask_rgb = np.array(Image.open(self.mask_paths[index]).convert("RGB"))
        green_channel = mask_rgb[:, :, 1]
        mask = (green_channel > 127).astype(np.float32)
        
        edge = np.array(Image.open(self.edge_paths[index]).convert("L"))
        edge = (edge > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, masks=[mask, edge])
            img = augmented['image']
            mask = augmented['masks'][0]
            edge = augmented['masks'][1]
        
        return img, mask.unsqueeze(0), edge.unsqueeze(0)

