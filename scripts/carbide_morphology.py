import os
import yaml
import argparse
import sys 
from pathlib import Path  
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.morphologies import get_percentage_list,plot_comparison_figure,write_percentage_orientation_to_file,get_area_percentage

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Unet', choices=['Unet', 'Segformer', 'MatSegNet','FPN'])
args = parser.parse_args()


CONFIGURE_REGISTRY={
'Unet':"Unet.yaml",
'Segformer':"Segformer.yaml",
'MatSegNet':"MatSegNet.yaml",
'FPN':"FPN.yaml",
}

configs_dir = os.path.abspath(os.path.join(project_root,  'configs'))
print(f"Starting training {args.model} model ... ")
CONFIG_FILE_PATH = os.path.join(configs_dir,CONFIGURE_REGISTRY[args.model])
print(f"[*] Loading configuration from: {CONFIG_FILE_PATH}")
with open(CONFIG_FILE_PATH, 'r') as f:
    config = yaml.safe_load(f)


images_saved_dir=os.path.join(project_root,'outputs',config['paths']['accuracy_result_name'],config['paths']['carbide_morphology_path']) 
bainite = os.path.join(project_root,'outputs',config['paths']['accuracy_result_name'],'Predictions','bainite') 
martensite =os.path.join(project_root,'outputs',config['paths']['accuracy_result_name'],'Predictions','martensite')  

bainite_list = [os.path.join(bainite,i) for i in os.listdir(bainite)]
martensite_list = [os.path.join(martensite,i) for i in os.listdir(martensite)]

print(f'there are {len(bainite_list)} bainite images, and {len(martensite_list)} martensite images')



bain_area_percentage_list=get_area_percentage(bainite_list)
mar_area_percentage_list=get_area_percentage(martensite_list)


bainite_k_list=get_percentage_list(bainite_list,images_saved_dir,name="Lower_Bainite")
martensite_k_list=get_percentage_list(martensite_list,images_saved_dir,name="Tempered_Martensite")

plot_comparison_figure(bain_area_percentage_list, mar_area_percentage_list, bainite_k_list, martensite_k_list, ylabel_a='Carbide Volume Fraction', ylabel_b='k',save_path=os.path.join(Path(images_saved_dir).parent,"Percentage_and_orientation_percentage.png"))



data_to_write = {
    'bain_area_percentage_list': bain_area_percentage_list,
    'mar_area_percentage_list': mar_area_percentage_list,
    'bainite_k_list': bainite_k_list,
    'martensite_k_list': martensite_k_list
} 

write_percentage_orientation_to_file(os.path.join(Path(images_saved_dir).parent,f"Percentage_Orientation_{args.model}.csv"),data_to_write)

print(f"File 'Percentage_Orientation_v2_{args.model}.csv' has been written successfully.")
