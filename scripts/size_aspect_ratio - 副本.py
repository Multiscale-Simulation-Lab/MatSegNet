from pathlib import Path  
import os
import numpy as np
import pandas as pd
import shutil
import cv2
import sys
import csv
import argparse
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.sizes_and_aspect_ratios import get_image_size_from_df,load_images_from_folder,merge_tiles_to_image,resize_images,calculate_precipitate_sizes,plot_size_distribution,process_steel_type,aspect_ratio_plot,print_sizes_aspect_ratios_to_csv

configs_dir = os.path.abspath(os.path.join(project_root,  'configs'))

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Unet', choices=['Unet', 'Segformer', 'MatSegNet','FPN'])
args = parser.parse_args()
model_name=args.model
CONFIGURE_REGISTRY={
'Unet':"Unet.yaml",
'Segformer':"Segformer.yaml",
'MatSegNet':"MatSegNet.yaml",
'FPN':"FPN.yaml"
}
print(f"Starting analyzing sizes and aspect ratios for predictions from  {model_name} model ... ")

CONFIG_FILE_PATH = os.path.join(configs_dir,CONFIGURE_REGISTRY[model_name])

print(f"[*] Loading configuration from: {CONFIG_FILE_PATH}")

with open(CONFIG_FILE_PATH, 'r') as f:
    config = yaml.safe_load(f)  



HOME_DIR=Path(os.path.join(project_root,"outputs",config['paths']['accuracy_result_name']))
PREDICTIONS_DIR = Path(os.path.join(HOME_DIR,"Predictions") )
INPUT_MARTENSITE_PATH = Path(os.path.join(PREDICTIONS_DIR,"martensite")  )
INPUT_BAINITE_PATH = Path(os.path.join(PREDICTIONS_DIR,"bainite") )
TEXT_FILE_PATH = Path(os.path.join(project_root,"data","SEM_images"))
MERGED_IMAGES_PATH = Path(os.path.join(PREDICTIONS_DIR,"big_merged_images"))
OUTPUT_DIR = Path(os.path.join(PREDICTIONS_DIR,"aspect_ratio_analysis"))


STEEL_TYPE_PATHS = {
    'N5325BAINITE': INPUT_BAINITE_PATH,
    'N5440MARTENSITE': INPUT_MARTENSITE_PATH
}
STEEL_TYPES = {
    "N5325BAINITE": Path(os.path.join(MERGED_IMAGES_PATH,"N5325BAINITE")) ,
    "N5440MARTENSITE": Path(os.path.join(MERGED_IMAGES_PATH,"N5440MARTENSITE"))
}

TILE_HEIGHT = config['post_processing']['tile_height']
TILE_WIDTH =config['post_processing']['tile_width']
LOWER_BOUND_COLOR = np.array(config['post_processing']['lower_bound_color'])
UPPER_BOUND_COLOR = np.array(config['post_processing']['upper_bound_color'])
NM_PER_PIXEL_BASE_MAG = config['post_processing']['nm_per_pixel_base_mag']
MIN_PRECIPITATE_PIXEL_AREA = config['post_processing']['min_precipitate_pixel_area']
EXCLUDED_MAGNIFICATION = config['post_processing']['excluded_magnification']
FIGURE_DPI = config['post_processing']['figure_dpi']
CONTOUR_AREA_RANGE = (config['post_processing']['contour_area_min'],config['post_processing']['contour_area_max'])
BOX_COLOR_BGR = config['post_processing']['box_color_bgr']


dataframe_dict = {}
original_dataframe = {}
for file_path in TEXT_FILE_PATH.glob('*_resized.txt'):
    df = pd.read_csv(file_path, header=None, delimiter=r"\s+")
    name = file_path.name.split('_')[0]
    dataframe_dict[name] = get_image_size_from_df(name, df)
    
    

for file_path in TEXT_FILE_PATH.glob('*_without_resize.txt'):
    df = pd.read_csv(file_path, header=None, delimiter=r"\s+")
    name = "".join(file_path.name.split('_')[0])
    print(name)
    original_dataframe[name] = get_image_size_from_df(name, df)


all_resized_images = {}
for steel_name, folder_path in STEEL_TYPE_PATHS.items():
    print(f"\nProcessing steel type: {steel_name}")
    print(f"\folder path type: {folder_path}")
    image_tiles = load_images_from_folder(folder_path)
    merged_images = merge_tiles_to_image(
        image_tiles, dataframe_dict.get(steel_name, {}), TILE_HEIGHT, TILE_WIDTH
    )
    all_resized_images[steel_name] = resize_images(
        merged_images, original_dataframe.get(steel_name, {})
    )
    
    
 
if MERGED_IMAGES_PATH.exists():
    shutil.rmtree(MERGED_IMAGES_PATH)
MERGED_IMAGES_PATH.mkdir(parents=True)

for steel_name, images in all_resized_images.items():
    steel_output_dir = MERGED_IMAGES_PATH / steel_name
    steel_output_dir.mkdir()
    for image_name, img_data in images.items():
 
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        save_path = steel_output_dir / f"{image_name}.png"
        cv2.imwrite(str(save_path), img_bgr)
        
        
merged_size_list = {}
for steel_name, images in all_resized_images.items():
    all_sizes = []
    for image_name, img_data in images.items():
        print(f"Analyzing precipitates in: {image_name}")
        sizes = calculate_precipitate_sizes(image_name, img_data,EXCLUDED_MAGNIFICATION,LOWER_BOUND_COLOR,UPPER_BOUND_COLOR,MIN_PRECIPITATE_PIXEL_AREA,NM_PER_PIXEL_BASE_MAG)
        all_sizes.extend(sizes)
    merged_size_list[steel_name] = np.array(all_sizes)
    

plot_size_distribution(merged_size_list,HOME_DIR,config['post_processing']['total_area_lb'],config['post_processing']['total_area_tm'])


if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)


aspect_ratio_data = {}
for name, path in STEEL_TYPES.items():
    aspect_ratio_data[name] = process_steel_type(name, path,OUTPUT_DIR,EXCLUDED_MAGNIFICATION,CONTOUR_AREA_RANGE,BOX_COLOR_BGR)
    
    
aspect_ratio_plot(aspect_ratio_data,HOME_DIR)



print_sizes_aspect_ratios_to_csv(merged_size_list,aspect_ratio_data,HOME_DIR,model_name)