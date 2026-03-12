import argparse
import os
import yaml
import torch
import sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
configs_dir = os.path.abspath(os.path.join(project_root,  'configs'))
from src.training import Trainer,calculate_segmentation_metrics
from models import Segformer, MatSegNet, Unet,FPN
from src.visualization import generate_merged_images,get_iou_list,save_iou_results_to_file,plot_iou_scatter,plot_training_history,save_training_history,final_evaluation,save_all_predictions,write_test_results_to_file,save_error_map_visualizations,predict_and_save_folder

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Unet', choices=['Unet', 'Segformer', 'MatSegNet','FPN'])
parser.add_argument('--type', type=str, default='best', choices=['best', 'newest'])
args = parser.parse_args()



CONFIG_REGISTRY = {
    "Unet": "Unet.yaml",
    "Segformer": "Segformer.yaml",    
    "MatSegNet": "MatSegNet.yaml",
    "FPN": "FPN.yaml"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name=args.model

print(f"Starting visualizing results of {args.model} model ... ")
CONFIG_FILE_PATH = os.path.join(configs_dir,CONFIG_REGISTRY[args.model])

print(f"[*] Loading configuration from: {CONFIG_FILE_PATH}")

with open(CONFIG_FILE_PATH, 'r') as f:
    config = yaml.safe_load(f)  

data_initializaion=Trainer(CONFIG_FILE_PATH,args.type,"f1_score")    
prediction_folder=os.path.join(project_root,"outputs",config['paths']['accuracy_result_name'])
generate_merged_images(model_name, data_initializaion.model,data_initializaion.test_loader,prediction_folder,"test_set")



results_train = get_iou_list(data_initializaion.train_loader, data_initializaion.model,model_name)
results_validation = get_iou_list(data_initializaion.validation_loader, data_initializaion.model,model_name)
results_test = get_iou_list(data_initializaion.test_loader, data_initializaion.model,model_name)

save_iou_results_to_file(
os.path.join(prediction_folder,f"Total_IoU_Scores_{args.type}.txt"), 
results_train, 
results_validation, 
results_test)

print(f"IoU results have been saved to {prediction_folder}/Total_IoU_Scores_{args.type}.txt")

plot_iou_scatter(
results_train, 
results_validation, 
results_test,
data_initializaion.train_loader,     
data_initializaion.validation_loader, 
data_initializaion.test_loader ,    
os.path.join(prediction_folder,f"IoU_Scatter_Plot_{args.type}.png"))

plot_training_history(data_initializaion.state.history,os.path.join(prediction_folder,f"Training_History_Plot_{args.type}.png"))

print("All results have been saved and plotted.")

size_train = len(data_initializaion.train_loader.dataset)
size_validation = len(data_initializaion.validation_loader.dataset)
size_test = len(data_initializaion.test_loader.dataset)

print('\n--- Dataset Sizes ---')
print(f'Train size: {size_train}, Validation size: {size_validation}, Test size: {size_test}')

save_training_history(data_initializaion.state.history,prediction_folder)


print("Evaluating final model on the Test Set...")

test_results = final_evaluation(data_initializaion.test_loader, data_initializaion.model, model_name, device)
write_test_results_to_file(test_results, prediction_folder, model_name, filename="test_results.txt")

print(f"\n--- Test Set Results ---")
print(f"Test set accuracy: {test_results['accuracy']:.4f}")
print(f"Test set recall: {test_results['recall']:.4f}")
print(f"Test set precision: {test_results['precision']:.4f}")
print(f"Test set F1 score: {test_results['f1_score']:.4f}")


saved_path_train = os.path.join(prediction_folder,'Predictions','train')
saved_path_validation = os.path.join(prediction_folder,'Predictions','validation')
saved_path_test = os.path.join(prediction_folder,'Predictions','test')

print("\n--- Starting to Save All Model Predictions to Disk ---")

save_all_predictions(loader=data_initializaion.train_loader, model=data_initializaion.model, model_name=model_name,device=device, output_path=saved_path_train,name_prefix='train')

save_all_predictions(loader=data_initializaion.validation_loader,model=data_initializaion.model, model_name=model_name,
device=device, output_path=saved_path_validation,name_prefix='validation')

save_all_predictions(loader=data_initializaion.test_loader,model=data_initializaion.model, model_name=model_name,device=device, output_path=saved_path_test,name_prefix='test')



print("\n--- All prediction images have been saved successfully. ---")
path_train_superposition = os.path.join(prediction_folder,'Predictions','train_superposition')
path_validation_superposition= os.path.join(prediction_folder,'Predictions','validation_superposition')
path_test_superposition = os.path.join(prediction_folder,'Predictions','test_superposition')
print("\n--- Starting to Save All Error Map Visualizations to Disk ---")


print("\n--- Starting to Save All Error Map Visualizations to Disk ---")

save_error_map_visualizations(loader=data_initializaion.train_loader, model=data_initializaion.model, model_name=model_name,device=device, output_path=path_train_superposition,name_prefix='train')

save_error_map_visualizations(loader=data_initializaion.validation_loader,model=data_initializaion.model,model_name=model_name,device=device,output_path=path_validation_superposition,name_prefix='validation')

save_error_map_visualizations(loader=data_initializaion.test_loader, model=data_initializaion.model, model_name=model_name,device=device, output_path=path_test_superposition,name_prefix='test')

print("\n--- All error map images have been saved successfully. ---")


input_bainite = os.path.join(project_root,'data','datasets','bainite_set','original') 
input_tempered_martensite= os.path.join(project_root,'data','datasets','martensite_set','original') 
output_bainite = os.path.join(prediction_folder,'Predictions','bainite')
output_martensite= os.path.join(prediction_folder,'Predictions','martensite')

predict_and_save_folder(model=data_initializaion.model,model_name=model_name, device=device,input_folder=input_bainite,output_folder=output_bainite,input_resize_dim=(data_initializaion.data_cfg['img_height'], data_initializaion.data_cfg['img_width']) )

predict_and_save_folder(model=data_initializaion.model,model_name=model_name,device=device,input_folder=input_tempered_martensite,output_folder=output_martensite,input_resize_dim=(data_initializaion.data_cfg['img_height'], data_initializaion.data_cfg['img_width']))

print(f"\n--- All bainite and martensite map images have been saved successfully to {output_bainite} and {output_martensite}. ---")
