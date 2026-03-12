import os
import shutil
import torch
import sys 
current_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(script_directory)
sys.path.append(project_root)
from src.preprocessing import save,load_images_names_lists_from_folder,create_dir,create_dirs,split_and_save


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = os.path.join(project_root, 'data','datasets')
image_dir = os.path.join(project_root, 'data', 'SEM_images')


if  os.path.exists(base_dir):  
    shutil. rmtree(base_dir)

# Output sets
sets = ['training_set', 'validation_set', 'test_set']
subdirs = ['mask', 'original', 'edge']
set_dirs = {s: {sub: os.path.join(base_dir, s, sub) for sub in subdirs} for s in sets}

# Phase-specific sets
phases = ['bainite_set', 'martensite_set']
phase_dirs = {p: {sub: os.path.join(base_dir, p, sub) for sub in subdirs} for p in phases}

# Create all directories
create_dirs([base_dir, image_dir])
for dirs in set_dirs.values():
    create_dirs(dirs.values())
for dirs in phase_dirs.values():
    create_dirs(dirs.values())

# === Raw image source paths ===
image_sources = {
    'N5_440_MARTENSITE': {
        'original': 'original_N5_440_MARTENSITE/cropped',
        'mask': 'mask_N5_440_MARTENSITE/cropped',
        'edge': 'edge_N5_440_MARTENSITE/cropped',
    },
    'N5_325_BAINITE': {
        'original': 'original_N5_325_BAINITE/cropped',
        'mask': 'mask_N5_325_BAINITE/cropped',
        'edge': 'edge_N5_325_BAINITE/cropped',
    },
    '42CrMo4_BAINITE': {
        'original': 'original_42CrMo4_BAINITE/cropped',
        'mask': 'mask_42CrMo4_BAINITE/cropped',
        'edge': 'edge_42CrMo4_BAINITE/cropped',
    },
}

# === Load images ===
datasets = {}
name_dict={}

for key, paths in image_sources.items():
    datasets[key] = {
        'original': load_images_names_lists_from_folder(os.path.join(image_dir, paths['original']))[1],
        'mask': load_images_names_lists_from_folder(os.path.join(image_dir, paths['mask']))[1],
        'edge': load_images_names_lists_from_folder(os.path.join(image_dir, paths['edge']))[1],
    }
    
for key, paths in image_sources.items():
    name_dict[key] = {
        'original': load_images_names_lists_from_folder(os.path.join(image_dir, paths['original']))[0],
        'mask': load_images_names_lists_from_folder(os.path.join(image_dir, paths['mask']))[0],
        'edge': load_images_names_lists_from_folder(os.path.join(image_dir, paths['edge']))[0],
    }
    
    
# === Print dataset summary ===
for name, ds in datasets.items():
    print(f"{name} → images: {len(ds['original'])}, masks: {len(ds['mask'])}, edges: {len(ds['edge'])}")



split_and_save(datasets['N5_440_MARTENSITE']['original'], datasets['N5_440_MARTENSITE']['mask'], datasets['N5_440_MARTENSITE']['edge'],name_dict['N5_440_MARTENSITE']['original'], name_dict['N5_440_MARTENSITE']['mask'], name_dict['N5_440_MARTENSITE']['edge'],
               'martensite', base_dir, base_dir)
               
split_and_save(datasets['N5_325_BAINITE']['original'], datasets['N5_325_BAINITE']['mask'], datasets['N5_325_BAINITE']['edge'],name_dict['N5_325_BAINITE']['original'], name_dict['N5_325_BAINITE']['mask'], name_dict['N5_325_BAINITE']['edge'],
               'bainite', base_dir, base_dir)
               
               
split_and_save(datasets['42CrMo4_BAINITE']['original'], datasets['42CrMo4_BAINITE']['mask'], datasets['42CrMo4_BAINITE']['edge'],name_dict['42CrMo4_BAINITE']['original'], name_dict['42CrMo4_BAINITE']['mask'], name_dict['42CrMo4_BAINITE']['edge'],
               'bainite', base_dir, base_dir)
               
print("folders for the training set, validation set, test set, bainite set, martensite set has all been generated, congratulations!")