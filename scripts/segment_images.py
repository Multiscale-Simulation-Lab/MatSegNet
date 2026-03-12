import os
import shutil
import sys
import yaml 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.preprocessing import load_images_from_folder,save_resized_images_to_folder,cropSave,mask_to_edge_from_rgb_dict




configs_dir = os.path.abspath(os.path.join(project_root,  'configs'))
CONFIG_FILE_PATH = os.path.join(configs_dir,"preprocess.yaml")

print(f"[*] Loading configuration from: {CONFIG_FILE_PATH}")

with open(CONFIG_FILE_PATH, 'r') as f:
    config = yaml.safe_load(f)  

desired_magnification=config["desired_magnification"]
height = config["image_height"]
weight = config["image_weight"]
input_dir = os.path.join(project_root,"data", "SEM_images")


def process_images():
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    print(f"Starting processing on directory: {input_dir}")
    print(f"Output will be saved to: {input_dir}")
    print("-" * 30)

    for it in os.scandir(input_dir):
   
        if it.is_dir() and it.name[-3:] =='ITE':
            steel_dir=os.path.join(input_dir, it.name)
            

            if it.name[0:4]=='mask':
                images=load_images_from_folder(steel_dir,desired_magnification,type=None,write=False)
            elif it.name[:8] =='original':
                images=load_images_from_folder(steel_dir,desired_magnification,type=it.name,write=True)
            elif it.name[0:4]=='edge':
                images=load_images_from_folder(steel_dir,desired_magnification,type=None,write=False)
                images=mask_to_edge_from_rgb_dict(images) 
            else:
                raise ValueError(f"Unrecognized folder name: {it.name}. Must start with 'mask', 'original', or 'edge'.")

            size_changed_dir=os.path.join(steel_dir,'size_changed')
            if  os.path.exists(size_changed_dir):
                shutil.rmtree(size_changed_dir)
            os.mkdir(size_changed_dir) 

            save_resized_images_to_folder(images,size_changed_dir)
            cropped_dir=os.path.join(steel_dir,'cropped')

            if  os.path.exists(cropped_dir):
                shutil.rmtree(cropped_dir)
            os.mkdir(cropped_dir) 


            image_name=(''.join(it.name.split('_')[1:]))
    
            
            cropSave(images,input_dir,str(image_name), height, weight, cropped_dir,write=True)
            


if __name__ == "__main__":

    process_images()
    print("Resized and Cropped Images have been generated !")
