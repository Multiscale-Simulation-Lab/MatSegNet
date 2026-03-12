import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_images_from_folder(folder,desired_magnification,type=None,write=False):
    
    if write==True:
        output_file=os.path.join(Path(folder).parent,
        f"{''.join(type.split('_')[1:])}_without_resize.txt")
        with open(output_file, 'w') as f:
            f.write('')

    images = {}
    for filename in os.listdir(folder):
        if filename[-3:] in ['png','jpg','gif','svg','peg']:
            magnification=(int(filename.split('X')[0].split('-')[1]))
            change_ratio=desired_magnification/magnification
            img = cv2.imread(os.path.join(folder,filename))
            [y_dim,x_dim]=img.shape[:2]
            if write==True:
              
                with open(output_file,"a") as f:  
                    print('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(filename).split('.png')[0],y_dim,x_dim))
                    f.write('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(filename).split('.png')[0],y_dim,x_dim))        
       
           
            x_dim=int(x_dim*change_ratio)
            y_dim=int(y_dim*change_ratio)
            img = cv2.resize(img,(x_dim,y_dim))
            [x_dim,y_dim]=img.shape[:2]
        
 
            if img is not None:
                images[str(filename)]=(img)
    return images
        
        

def save_resized_images_to_folder(images, saved_path):
    os.makedirs(saved_path, exist_ok=True)
    
    for name, img in images.items():
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()

        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        img = img - img.min()
        img = img / (img.max() + 1e-8)

    
        save_path = os.path.join(saved_path, name)
        plt.imsave(save_path, img, cmap='gray')



        
def save(img_array, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    for index in range(len(img_array)):
        save_path = os.path.join(path, name[index])
        cv2.imwrite(save_path, img_array[index])
        

def cropSave(images,current_path, type,h, w, saved_path,write=False):
    if write==True:
        output_file=os.path.join(current_path,"{}_resized.txt".format(type))
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'w') as f:
            f.write('')
    if  os.path.exists(saved_path):
        shutil.rmtree(saved_path)  
    os.mkdir(saved_path)
    os.chdir(saved_path)
    
    for i in images.keys():
        [y_dim,x_dim]=images[i].shape[:2]
        if write==True:
            print('Output Files: ',output_file)
            with open(output_file,"a") as f:  
                print('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(i).split('.png')[0],y_dim,x_dim))
                f.write('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(i).split('.png')[0],y_dim,x_dim))        
        for y in range(int(np.ceil(y_dim / h))):
            for x in range(int(np.ceil(x_dim/ w))):
                if (y ==np.ceil(y_dim / h)-1) and (x ==np.ceil(x_dim / w)-1):
                    print('The last y is:',y,'The last x is :',x)
                    cropped_img = images[i][images[i].shape[0]-h:images[i].shape[0],  images[i].shape[1]-w:images[i].shape[1]]
                    image_name=(type+str(i) + str(y) + str(x) +'.png')
                    cv2.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)
                    continue
                if y ==np.ceil(y_dim / h)-1:
                    cropped_img = images[i][images[i].shape[0]-h:images[i].shape[0], x*w:(x+1)*w]
                    image_name=(type+str(i) + str(y) + str(x) +'.png')
                    cv2.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)
                    continue
                if x ==np.ceil(x_dim / w)-1:
                    cropped_img = images[i][y*h:(y+1)*h, images[i].shape[1]-w:images[i].shape[1]]
                    image_name=(type+str(i) + str(y) + str(x) +'.png')
                    cv2.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)
                    continue
                cropped_img = images[i][y*h:(y+1)*h, x*w:(x+1)*w]
                image_name=(type+str(i) + str(y) + str(x) +'.png')
          
                cv2.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)


    
def mask_to_edge_from_rgb_dict(mask_rgb_dict, kernel_size=3, low_thresh=50, high_thresh=150):
    edge_tensor_dict = {}

    for name, rgb_img in mask_rgb_dict.items():
        # 1. Convert to grayscale
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        gray[gray > 127] = 255
        gray[gray < 127] = 0
      
        # 2. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # 3. Perform Canny edge detection
        edge = cv2.Canny(blurred, threshold1=low_thresh, threshold2=high_thresh)

        # 4. FIND AND FILTER CONTOURS TO REMOVE INTERNAL EDGES
        #    cv2.RETR_EXTERNAL retrieves only the outermost contours.
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new black image to draw only the outer edges
        outer_edge_mask = np.zeros_like(edge)
        # Increase the thickness by changing the last parameter
        cv2.drawContours(outer_edge_mask, contours, -1, (255), 2) # <-- THICKNESS CHANGED HERE

    
        # 5. Normalize and convert to tensor (1, H, W)
        normalized_edge = outer_edge_mask.astype(np.float32) / 255.0
        edge_tensor = torch.from_numpy(normalized_edge).unsqueeze(0)

        edge_tensor_dict[name] = edge_tensor[0].numpy() * 255

    return edge_tensor_dict
    
    
def load_images_names_lists_from_folder(folder):
    images = []
    names=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            names.append(filename)
    return names,images
    
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



    
def create_dirs(dirs):
    for d in dirs:
        create_dir(d)
        
        
def split_and_save(original, mask, edge,
                   name_original, name_mask, name_edge,
                   base_name, base_dir, structure_dir):
    
    data = list(zip(original, mask, edge))
    name_data = list(zip(name_original, name_mask, name_edge))

    train, rem = train_test_split(data, train_size=0.7, random_state=69)
    val, test = train_test_split(rem, test_size=0.5, random_state=69)

    train_name, rem_name = train_test_split(name_data, train_size=0.7, random_state=69)
    val_name, test_name = train_test_split(rem_name, test_size=0.5, random_state=69)

    for split_name, split_data, split_names in zip(
            ['training_set', 'validation_set', 'test_set'],
            [train, val, test],
            [train_name, val_name, test_name]):

        x, y, e = zip(*split_data)
        x_names, y_names, e_names = zip(*split_names)

        save(x, os.path.join(base_dir, split_name, 'original'), x_names)
        save(y, os.path.join(base_dir, split_name, 'mask'), y_names)
        save(e, os.path.join(base_dir, split_name, 'edge'), e_names)

        if 'martensite' in base_name.lower():
            phase = 'martensite_set'
        elif 'bainite' in base_name.lower():
            phase = 'bainite_set'
        else:
            raise ValueError("Error: please specify the steel type (must include 'martensite' or 'bainite' in base_name)")

        save(x, os.path.join(structure_dir, phase, 'original'), x_names)
        save(y, os.path.join(structure_dir, phase, 'mask'), y_names)
        save(e, os.path.join(structure_dir, phase, 'edge'), e_names)
