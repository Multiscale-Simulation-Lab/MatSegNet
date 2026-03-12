import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models import Segformer, MatSegNet, Unet, FPN
import yaml
from src.load_data import get_data_paths,get_sorted_image_mask_lists,get_sorted_image_mask_edge_lists,create_dataloaders,create_datasets,create_edge_datasets,get_data_paths
from src.checkpoints  import TrainingState, load_checkpoint,save_checkpoint
import torch
import segmentation_models_pytorch as smp

def calculate_segmentation_metrics(preds, masks, positive_class=1):
    preds = preds.long()
    masks = masks.long()
    preds_positive = (preds == positive_class).float()
    masks_positive = (masks == positive_class).float()
    tp = torch.sum(preds_positive * masks_positive).item()
    fp = torch.sum(preds_positive * (1 - masks_positive)).item()
    fn = torch.sum((1 - preds_positive) * masks_positive).item()
    tn = torch.sum((1 - preds_positive) * (1 - masks_positive)).item()
    return {'tp': tp, 'fp': fp, 'fn': fn,'tn':tn}

def compute_metrics_from_stats(stats: dict) -> dict:
    tp = stats.get('tp', 0)
    fp = stats.get('fp', 0)
    fn = stats.get('fn', 0)
    tn = stats.get('tn', 0)
    total_pixels = stats.get('total_pixels', 0)
    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    recall = tp / (tp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        "precision":precision
    }



def train_one_epoch_Unet(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}
    
    progress_bar = tqdm(loader, desc="Training")
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        batch_stats = calculate_segmentation_metrics(preds, masks)
        epoch_stats['tp'] += batch_stats['tp']
        epoch_stats['fp'] += batch_stats['fp']
        epoch_stats['fn'] += batch_stats['fn']
        
        progress_bar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy


def validate_one_epoch_Unet(model, loader, loss_fn, device):
    print("Checking F1-Score and loss on validation set...")
    model.eval()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}

    progress_bar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            batch_stats = calculate_segmentation_metrics(preds, masks)
            epoch_stats['tp'] += batch_stats['tp']
            epoch_stats['fp'] += batch_stats['fp']
            epoch_stats['fn'] += batch_stats['fn']

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy


def train_one_epoch_MatSegNet(model, loader, optimizer, loss_fn_mask, loss_fn_edge, scaler, loss_weights, device):
    model.train()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}

    progress_bar = tqdm(loader, leave=True, desc="Training")
    for images, masks, edges in progress_bar:
        images = images.to(device)
        masks = masks.to(device, dtype=torch.float32)
        edges = edges.to(device, dtype=torch.float32)

        with torch.amp.autocast('cuda'):
            logits_mask, logits_edge = model(images)
            loss_m = loss_fn_mask(logits_mask, masks)
            loss_e = loss_fn_edge(logits_edge, edges)
            total_batch_loss = (loss_weights['mask'] * loss_m) + (loss_weights['edge'] * loss_e)

        optimizer.zero_grad()
        scaler.scale(total_batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_batch_loss.item()* images.size(0)
        
        preds_mask = (torch.sigmoid(logits_mask) > 0.5).float()
        batch_stats = calculate_segmentation_metrics(preds_mask, masks)
        epoch_stats['tp'] += batch_stats['tp']
        epoch_stats['fp'] += batch_stats['fp']
        epoch_stats['fn'] += batch_stats['fn']

        progress_bar.set_postfix(loss=total_batch_loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy

def validate_one_epoch_MatSegNet(model, loader, loss_fn_mask, loss_fn_edge, loss_weights, device):
    print("Checking F1-Score and loss on validation set...")
    model.eval()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}

    progress_bar = tqdm(loader, leave=True, desc="Validation")
    with torch.no_grad():
        for images, masks, edges in loader:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.float32)
            edges = edges.to(device, dtype=torch.float32)
            
            logits_mask, logits_edge = model(images)
            loss_m = loss_fn_mask(logits_mask, masks)
            loss_e = loss_fn_edge(logits_edge, edges)
            batch_loss = (loss_weights['mask'] * loss_m) + (loss_weights['edge'] * loss_e)
            running_loss += batch_loss.item()*images.size(0)
            
            preds_mask = (torch.sigmoid(logits_mask) > 0.5).float()
            batch_stats = calculate_segmentation_metrics(preds_mask, masks)
            epoch_stats['tp'] += batch_stats['tp']
            epoch_stats['fp'] += batch_stats['fp']
            epoch_stats['fn'] += batch_stats['fn']

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy


def train_one_epoch_Segformer(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}
    
    progress_bar = tqdm(loader, desc="Training")
    for images,masks in progress_bar:
        images =images.to(device)
        masks = masks.squeeze(1).long().to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            logits = outputs.logits 
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        running_loss += loss.item()* images.size(0)
        
        upsampled_logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        predicted_mask = upsampled_logits.argmax(dim=1)
        batch_stats = calculate_segmentation_metrics(predicted_mask, masks)
        epoch_stats['tp'] += batch_stats['tp']
        epoch_stats['fp'] += batch_stats['fp']
        epoch_stats['fn'] += batch_stats['fn']

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy

def validate_one_epoch_Segformer(model, loader, device):
    model.eval()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}

    progress_bar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images,masks in progress_bar:
            images=images.to(device)
            masks=masks.squeeze(1).long().to(device)
           
            outputs = model(pixel_values=images, labels=masks)
            running_loss += outputs.loss.item()* images.size(0)

            upsampled_logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            predicted_mask = upsampled_logits.argmax(dim=1)
            batch_stats = calculate_segmentation_metrics(predicted_mask, masks)
            epoch_stats['tp'] += batch_stats['tp']
            epoch_stats['fp'] += batch_stats['fp']
            epoch_stats['fn'] += batch_stats['fn']
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy



def train_one_epoch_FPN(model, loader, optimizer, loss_fn, scaler, device, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}
    optimizer.zero_grad()   
    
    progress_bar = tqdm(loader, desc="Training")


    use_amp = (device.type == 'cuda')
    
    for step, (images, masks) in enumerate(progress_bar):
        images, masks = images.to(device), masks.to(device)
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = loss_fn(outputs, masks)/ accumulation_steps 
            preds = (torch.sigmoid(outputs) > 0.5).float()
     
        
        
        scaler.scale(loss).backward() 
        
        
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        true_batch_loss = loss.item() * accumulation_steps
        running_loss += true_batch_loss * images.size(0)
        
        batch_stats = calculate_segmentation_metrics(preds, masks)
        epoch_stats['tp'] += batch_stats['tp']
        epoch_stats['fp'] += batch_stats['fp']
        epoch_stats['fn'] += batch_stats['fn']
        
        progress_bar.set_postfix(loss=true_batch_loss)
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy


def validate_one_epoch_FPN(model, loader, loss_fn, device):
    print("Checking F1-Score and loss on validation set...")
    model.eval()
    running_loss = 0.0
    epoch_stats = {'tp': 0, 'fp': 0, 'fn': 0}

    progress_bar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            batch_stats = calculate_segmentation_metrics(preds, masks)
            epoch_stats['tp'] += batch_stats['tp']
            epoch_stats['fp'] += batch_stats['fp']
            epoch_stats['fn'] += batch_stats['fn']

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = compute_metrics_from_stats(epoch_stats)['f1']
    epoch_recall = compute_metrics_from_stats(epoch_stats)['recall']
    epoch_accuracy = compute_metrics_from_stats(epoch_stats)['accuracy']  
    return epoch_loss, epoch_f1,epoch_recall,epoch_accuracy


############################################################################
 

def get_parameters_from_config(model_name):
    configs_dir = os.path.abspath(os.path.join(project_root,  'configs'))
    CONFIG_FILE_PATH = os.path.join(configs_dir,(model_name+".yaml"))
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = yaml.safe_load(f)  
    return config
    
TEACHER_MODEL_DICT={"MatSegNet":"best_matsegnet.pth",
"FPN":"best_FPN_efficientNetB4.pth",
"Segformer":"best_segformer.pth"}


def load_teacher_models(model_classes, device):
    teacher_models = {}
    for model_class in model_classes:
        model = MODEL_REGISTRY[model_class].get_model(n_classes=get_parameters_from_config(f"{model_class}")['num_classes']).to(device)
        checkpoint_path = os.path.join(project_root, "outputs", "checkpoints", TEACHER_MODEL_DICT[model_class])
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:

            model.load_state_dict(checkpoint)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        teacher_models[model_class] = model

    return teacher_models





def get_training_functions(model_name):
    if model_name.lower() == "unet":
        return train_one_epoch_Unet,validate_one_epoch_Unet
    elif model_name.lower() == "matsegnet":
        return train_one_epoch_MatSegNet, validate_one_epoch_MatSegNet
    elif model_name.lower() == "segformer":
        return train_one_epoch_Segformer, validate_one_epoch_Segformer
    elif model_name.lower() == "fpn":
        return train_one_epoch_FPN,validate_one_epoch_FPN
    else:
        raise ValueError(f"Unknown model name: {model_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



LOSS_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "DiceLoss": smp.losses.DiceLoss
}
MODEL_REGISTRY = {
    "Unet": Unet,
    "Segformer": Segformer,
    "MatSegNet": MatSegNet,
    "FPN": FPN
}
SCALER_REGISTRY = {
    "GradScaler": torch.amp.GradScaler,
    
}
OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam
}


CHECKPOINT_REGISTRY={
"best": "best_checkpoint_path",
"newest": "latest_checkpoint_path"
}

ENCODER_NAME_MAP = {
"Segformer": "segformer.encoder",
"Unet": "encoder",
"MatSegNet": "encoder" ,
"FPN": "encoder"
}
        
class Trainer:
    def __init__(self, config_path, which_checkpoint,primary_metric):
        print("--- Initializing Trainer ---")
        self.config_path = config_path
        self.which_checkpoint = which_checkpoint
        self.primary_metric=primary_metric
        
        self._load_config()
        self._setup_environment()
        self._prepare_data()
        self._build_components()
        print("--- Trainer Initialization Complete ---")

    def _load_config(self):
        print(f"[*] Loading configuration from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        for key, value in self.config.items():
            setattr(self, f"{key}_cfg", value)
    
    def _setup_environment(self):
        print("[*] Setting up environment (device and paths)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sub_checkpoint_dir = os.path.join("outputs", "checkpoints")
        self.checkpoint_path = os.path.join(project_root, sub_checkpoint_dir, self.paths_cfg[CHECKPOINT_REGISTRY[self.which_checkpoint]])
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.dataset_directory = os.path.join(project_root, "data", "datasets")
        print(f"    - Device set to: {self.device}")
        print(f"    - Checkpoint path: {self.checkpoint_path}")
        
        self.two_stage_cfg = self.config.get('two_stage_training', {})
        self.enable_two_stage = self.two_stage_cfg.get('enable', False)
        
    def _prepare_data(self):
        print("[*] Preparing data loaders...")
        paths = get_data_paths(self.dataset_directory)
        if self.model_name_cfg in ["MatSegNet"]:
            image_list_train, mask_list_train,edge_list_train=get_sorted_image_mask_edge_lists(paths['image_train'], paths['mask_train'], paths['edge_train'])
            image_list_val, mask_list_val,edge_list_val=get_sorted_image_mask_edge_lists(paths['image_validation'], paths['mask_validation'], paths['edge_validation'])
            image_list_test, mask_list_test,edge_list_test=get_sorted_image_mask_edge_lists(paths['image_test'], paths['mask_test'], paths['edge_test'])

            _, _,train_dataset, val_dataset, test_dataset = create_edge_datasets(
            image_list_train, mask_list_train,edge_list_train,
            image_list_val, mask_list_val,edge_list_val,
            image_list_test, mask_list_test,edge_list_test,
            self.data_cfg['img_height'], self.data_cfg['img_width'])

        else:
            image_list_train, mask_list_train = get_sorted_image_mask_lists(paths['image_train'], paths['mask_train'])
            image_list_val, mask_list_val = get_sorted_image_mask_lists(paths['image_validation'], paths['mask_validation'])
            image_list_test, mask_list_test = get_sorted_image_mask_lists(paths['image_test'], paths['mask_test'])
        

            _, _, train_dataset, val_dataset, test_dataset = create_datasets(
            image_list_train, mask_list_train, 
            image_list_val, mask_list_val, image_list_test, mask_list_test,
            self.data_cfg['img_height'], self.data_cfg['img_width'])
        
        self.train_loader, self.validation_loader, self.test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.train_cfg['batch_size'], num_workers=0, pin_memory=True
        )
        print(f"    - Data loaders created successfully.")
        
        
    def _build_components(self):
        print("[*] Building model and other components...")
        model_class = MODEL_REGISTRY[self.model_name_cfg]
        self.model = model_class.get_model(n_classes=self.config['num_classes'])
        self.model.to(self.device)
        
        loss_config = self.train_cfg['loss_fn']      

        if self.model_name_cfg in ["MatSegNet"]:
            
            print("[*] Building loss functions for MatSegNet...")
            
            self.loss_fn_mask = LOSS_REGISTRY[loss_config['mask']['name']](**(loss_config['mask'].get('params') or {}))
            self.loss_fn_edge = LOSS_REGISTRY[loss_config['edge']['name']](**(loss_config['edge'].get('params') or {}))
          #  self.loss_fn={"mask":self.loss_fn_mask,"edge":self.loss_fn_edge}
            print(f"    - Mask Loss: {loss_config['mask']['name']}")
            print(f"    - Edge Loss: {loss_config['edge']['name']}")
     
        elif self.model_name_cfg in ["Unet", "FPN"]:
            print("[*] Building loss function for Unet...")
            self.loss_fn = LOSS_REGISTRY[loss_config['name']](**(loss_config.get('params') or {}))
            print(f"    - Loss: {loss_config['name']}")

        
        scaler_config_name = self.train_cfg.get('scaler')

        self.scaler = SCALER_REGISTRY[scaler_config_name]("cuda") if scaler_config_name and self.device.type == 'cuda' else None
        
        optimizer_config = self.train_cfg['optimizer']
        self.optimizer = OPTIMIZER_REGISTRY[optimizer_config['name']](
            self.model.parameters(),
            lr=self.train_cfg['learning_rate'],
            weight_decay=self.train_cfg.get('weight_decay', 1e-4),
            **(optimizer_config.get('params') or {})
        )

        self.state = load_checkpoint(
            self.model, self.optimizer, self.scaler,
            self.checkpoint_path, self.device, primary_metric='f1_score'
        )
        print(f"    - Model '{self.model_name_cfg}' and components built.")
        
#
    def run(self):
        
        if self.enable_two_stage:
            print("\n--- Starting  Two-Stage Training Loop ---")
            self._run_two_stage()  
        else:
            print("\n--- Starting Standard Single-Stage Training Loop ---")
            self._run_single_stage()

    def _run_single_stage(self):
        print("Running single stage training...")
        train_fn, eval_fn = get_training_functions(self.model_name_cfg)
        start_epoch = self.state.epoch
        num_classes = self.config['num_classes']
        metric_name = self.state.primary_metric
        
        for epoch in range(start_epoch, self.train_cfg['num_epochs']+1):
            print(f"\n===== Epoch {epoch + 1} / {self.train_cfg['num_epochs']} =====")

            if self.model_name_cfg == "MatSegNet":
                train_loss, train_f1,train_recall,train_accuracy = train_fn(self.model, self.train_loader, self.optimizer, self.loss_fn_mask, self.loss_fn_edge, self.scaler, self.train_cfg['loss_weights'], self.device)
                val_loss, val_f1,val_recall,val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn_mask, self.loss_fn_edge, self.train_cfg['loss_weights'], self.device)
            elif self.model_name_cfg == "Unet":
                train_loss, train_f1,train_recall,train_accuracy = train_fn(self.model, self.train_loader, self.optimizer, self.loss_fn, self.device)
                val_loss, val_f1,val_recall,val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn, self.device)
            elif self.model_name_cfg == "Segformer":
                train_loss, train_f1,train_recall,train_accuracy = train_fn(self.model, self.train_loader, self.optimizer, self.scaler, self.device)
                val_loss, val_f1,val_recall,val_accuracy = eval_fn(self.model, self.validation_loader, self.device)
            elif self.model_name_cfg == "FPN":
                train_loss, train_f1,train_recall,train_accuracy = train_fn(self.model, self.train_loader, self.optimizer,  self.loss_fn,self.scaler, self.device)
                val_loss, val_f1,val_recall,val_accuracy = eval_fn(self.model, self.validation_loader,  self.loss_fn, self.device)
            
            self.state.epoch = epoch + 1
            self._log_and_save(train_loss, train_f1, train_recall, train_accuracy, val_loss, val_f1, val_recall, val_accuracy)
        
        print("\n--- Training Finished ---")

    def _run_two_stage(self):
        print("Running two stage training...")
        train_fn, eval_fn = get_training_functions(self.model_name_cfg)
        
        start_epoch = self.state.epoch
        metric_name = self.state.primary_metric
        print("====start_epoch====",start_epoch)

        encoder_prefix = ENCODER_NAME_MAP[self.model_name_cfg]
        print(f"[*] Identified encoder prefix for {self.model_name_cfg}: '{encoder_prefix}'")

        optimizer_config = self.train_cfg['optimizer']
        num_epochs_stage1 = self.two_stage_cfg.get('epochs_stage1', 30)
        num_epochs_stage2 = self.two_stage_cfg.get('epochs_stage2', 10)

 
        if start_epoch < num_epochs_stage1:
            print("\n" + "="*30)
            print("  ENTERING STAGE 1: TRAINING DECODER HEAD ONLY")
            print("="*30)

            for name, param in self.model.named_parameters():
                if name.startswith(encoder_prefix):
                    param.requires_grad = False
            
            lr_stage1 = self.two_stage_cfg.get('lr_stage1', 1e-4)
            optimizer_stage1 = OPTIMIZER_REGISTRY[optimizer_config['name']](
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=lr_stage1, **(optimizer_config.get('params') or {}))

            for epoch in range(start_epoch, num_epochs_stage1):
                print(f"\n===== Stage 1, Epoch {epoch + 1}/{num_epochs_stage1} =====")

                if self.model_name_cfg == "Unet":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage1, self.loss_fn, self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn, self.device)
                elif self.model_name_cfg == "MatSegNet":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage1, self.loss_fn_mask, self.loss_fn_edge, self.scaler, self.train_cfg['loss_weights'], self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn_mask, self.loss_fn_edge, self.train_cfg['loss_weights'], self.device)
                    
                elif self.model_name_cfg == "FPN":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage1, self.loss_fn,self.scaler,  self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn, self.device)
                    
                elif self.model_name_cfg == "Segformer":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage1, self.scaler, self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.device)
                
               

                
                self.state.epoch = epoch + 1
                self._log_and_save(train_loss, train_f1, train_recall, train_accuracy, val_loss, val_f1, val_recall, val_accuracy)
 
        start_epoch_stage2 = max(num_epochs_stage1, start_epoch)
        
        print("start_epoch_stage2",start_epoch_stage2)
        
        if start_epoch_stage2 < (num_epochs_stage1 + num_epochs_stage2+1):
            print("\n" + "="*30)
            print("  ENTERING STAGE 2: FINE-TUNING ENTIRE MODEL")
            print("="*30)

            for param in self.model.parameters():
                param.requires_grad = True
                
            lr_stage2 = self.two_stage_cfg.get('lr_stage2', 1e-5)
            optimizer_stage2 = OPTIMIZER_REGISTRY[optimizer_config['name']](
                self.model.parameters(), 
                lr=lr_stage2, **(optimizer_config.get('params') or {}))

     
            for epoch in range(start_epoch_stage2, num_epochs_stage1 + num_epochs_stage2):
                relative_epoch_stage2 = epoch - num_epochs_stage1 + 1
                print(f"\n===== Stage 2, Epoch {relative_epoch_stage2}/{num_epochs_stage2} (Total Epoch: {epoch + 1}) =====")

                if self.model_name_cfg == "Unet":
            
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage2, self.loss_fn, self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn, self.device)
                elif self.model_name_cfg == "MatSegNet":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage2, self.loss_fn_mask, self.loss_fn_edge, self.scaler, self.train_cfg['loss_weights'], self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn_mask, self.loss_fn_edge, self.train_cfg['loss_weights'], self.device)
                elif self.model_name_cfg == "FPN":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage2, self.loss_fn,  self.scaler,self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.loss_fn, self.device)  
                elif self.model_name_cfg == "Segformer":
                    train_loss, train_f1, train_recall, train_accuracy = train_fn(self.model, self.train_loader, optimizer_stage2, self.scaler, self.device)
                    val_loss, val_f1, val_recall, val_accuracy = eval_fn(self.model, self.validation_loader, self.device)
                
                    
                self.state.epoch = epoch + 1
                self._log_and_save(train_loss, train_f1, train_recall, train_accuracy, val_loss, val_f1, val_recall, val_accuracy)

        print("\n--- Two-Stage Training Finished ---")

    def _log_and_save(self, train_loss, train_f1, train_recall, train_accuracy, val_loss, val_f1, val_recall, val_accuracy):
        """Helper function to log history and save checkpoints."""
        metric_name = self.state.primary_metric

   
        self.state.history['train_loss'].append(train_loss)
        self.state.history[f'train_{metric_name}'].append(train_f1)
        self.state.history[f'train_acc'].append(train_accuracy)
        self.state.history[f'train_recall'].append(train_recall)
        
        self.state.history['val_loss'].append(val_loss)
        self.state.history[f'val_{metric_name}'].append(val_f1)
        self.state.history['val_acc'].append(val_accuracy)
        self.state.history[f'val_recall'].append(val_recall)
        
        self.state.last_val_metric = val_f1

        print(f"Epoch {self.state.epoch} Summary:")
        print(f"  Training Loss:    {train_loss:.4f} | Training F1-Score:    {train_f1:.4f}")
        print(f"  Training Accuracy:    {train_accuracy:.4f} | Training Recall:    {train_recall:.4f}")

        print(f"  Validation Loss:  {val_loss:.4f} | Validation F1-Score:  {val_f1:.4f}")
        print(f"  Validation Accuracy:  {val_accuracy:.4f} | Validation Recall:  {val_recall:.4f}")
        
        save_checkpoint(self.state, self.checkpoint_path)