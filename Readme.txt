Segmentation_Pipeline/
├── configs/                 
│   └── config.py
│         └── MatSegNet.yaml
│         └── Segformer.yaml
│         └── Unet.yaml
├── data/                     
│   ├── SEM_images/           
│   ├── datasets/              
│  		└──bainite_set
│		└──martensite_set
│		└──training_set
│		└──validation_set
│		└──test_set
├── models/   
│   ├──MatSegNet.py
│   ├──Segformer.py
│   ├──Unet.py
├── output/                  
│   ├── checkpoints/
│         └──best_matsegnet.pth
│         └──best_segformer.pth
│         └──best_unet_mobilenetv2.pth
│         └──matsegnet.pth
│         └──segformer.pth
│         └──unet_mobilenetv2.pth
│   └── accuracy_output/     
├── src/            
│   ├── datasets/
│         └──preprocessing.py
│         └──load_data.py
│         └──checkpoints.py
│         └──training.py
│         └──visualization.py
├── scripts/
│         └──python segment_images.py
│         └──python train_test_split.py
│         └──python train.py  --model MatSegNet
│         └──python visualize_results.py  --model MatSegNet
│         └──python carbide_morphology.py  --model MatSegNet
│         └──python size_aspect_ratio.py  --model MatSegNet

├── .gitignore              
├── requirements.txt          
└── README.md     