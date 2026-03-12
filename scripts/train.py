import argparse
import sys 
import os
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.training import Trainer


configs_dir = os.path.abspath(os.path.join(project_root,  'configs'))


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Unet', choices=['Unet', 'Segformer', 'MatSegNet','FPN'])
args = parser.parse_args()



CONFIGURE_REGISTRY={
'Unet'      :   "Unet.yaml",
'Segformer' :   "Segformer.yaml",
'MatSegNet' :   "MatSegNet.yaml",
'FPN'       :   "FPN.yaml"
}

print(f"Starting training {args.model} model ... ")
CONFIG_FILE_PATH = os.path.join(configs_dir,CONFIGURE_REGISTRY[args.model])
train_initializaion=Trainer(CONFIG_FILE_PATH,"newest","f1_score")
print(f"[*] Loading configuration from: {CONFIG_FILE_PATH}")
train_initializaion.run()

    
