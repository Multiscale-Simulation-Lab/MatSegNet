import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(img_height, img_width):
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

def get_val_test_transform(img_height, img_width):
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])