from pathlib import Path
import os
import torch


class MetricTracker:

    def __init__(self, metric_name: str = 'f1_score', initial_best_value: float = 0.0):
        self.metric_name = metric_name
        self.best_value = initial_best_value
    def update(self, current_value):
        if current_value > self.best_value:
            print(f"✨ New best {self.metric_name} is now: {current_value:.4f} (before, it was {self.best_value:.4f})")
            self.best_value = current_value
            return True
        return False
        
class TrainingState:

    def __init__(self, model, optimizer, scaler=None, primary_metric: str = 'f1_score'):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.epoch = 0
        self.history = {
            'train_loss': [], f'train_{primary_metric}': [], 'train_acc': [],'train_recall': [],
            'val_loss': [],f'val_{primary_metric}': [],'val_acc': [],'val_recall': []
        }
        self.metric_tracker = MetricTracker(metric_name=primary_metric)
        self.last_val_metric = 0.0
        self.primary_metric = primary_metric



def save_checkpoint(state: TrainingState, checkpoint_path: str):
    checkpoint_dir=Path(checkpoint_path).parent
    file_name=Path(checkpoint_path).name  
    checkpoint = {
        'epoch': state.epoch,
        'model_state_dict': state.model.state_dict(),
        'optimizer_state_dict': state.optimizer.state_dict(),
        'history': state.history,
        'best_metric_value': state.metric_tracker.best_value, 
        'metric_name': state.primary_metric
    }
    
    if state.scaler:
        checkpoint['scaler_state_dict'] = state.scaler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Latest checkpoint (Epoch {state.epoch}) saved to: {checkpoint_path}")

    if state.metric_tracker.update(state.last_val_metric):
        best_path = os.path.join(checkpoint_dir, f"best_{file_name}")
        print(f"  -> New best model found! Saving to {best_path}")
        torch.save(checkpoint, best_path)

    
def load_checkpoint(model, optimizer, scaler, checkpoint_path, device, primary_metric: str = 'f1_score'):

    state = TrainingState(model, optimizer, scaler, primary_metric)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint was not found at {checkpoint_path}, we will start from epoch 0.")
        return state

    print(f"Checkpoint is found: {checkpoint_path}, training is restarted...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state.model.load_state_dict(checkpoint['model_state_dict'])
    state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    state.epoch = checkpoint.get('epoch', 0)
    state.history = checkpoint.get('history', state.history)
    

    best_metric_value = checkpoint.get('best_metric_value', 0.0)
    metric_name = checkpoint.get('metric_name', primary_metric) 
    state.metric_tracker = MetricTracker(metric_name=metric_name, initial_best_value=best_metric_value)
    

    state.primary_metric = metric_name
    
    print(f"The best {state.metric_tracker.metric_name} from previous training: {state.metric_tracker.best_value:.4f}")

    if state.scaler and 'scaler_state_dict' in checkpoint:
        state.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("Scaler state has been successfully loaded.")
    
    print(f"Checkpoint loaded: {state.epoch } epochs have been trained")
    return state
    
    