# Deep Learning-based Image Segmentation Pipeline
## 1. Overview

This project provides a complete pipeline for training and evaluating four different deep learning models (FPN, SegFormer, UNet, and MatSegNet) for an image segmentation task. The framework covers the entire workflow, from data preprocessing (image cropping and dataset splitting) to model training, prediction, post-processing, and quantitative analysis of the segmentation results and secondary precipitates.

In our tests, the MatSegNet model demonstrated superior performance, surpassing other models on both `recall` and `precision` metrics.

## 2. Key Features

*   **Multi-Model Support**: Integrates four powerful segmentation models: FPN, SegFormer, UNet, and MatSegNet.
*   **End-to-End Workflow**: Offers a complete solution from raw images to final quantitative analysis.
*   **Automated Data Preparation**: Includes scripts for automated image cropping and splitting into training, validation, and test sets.
*   **Advanced Post-processing**: Supports prediction on tiled images, merging them back to their original size, and creating visual comparisons with the source images.
*   **Quantitative Morphological Analysis**: Performs a detailed analysis of the morphology of the predicted precipitates.

## 3. Model Architectures

This pipeline includes the following models:
*   **FPN**: FPN fuses multi-scale features through top-down pathways and lateral connections, enabling robust detection and segmentation across varying object scales.

*   **UNet**: A classic and widely-used convolutional neural network with an encoder-decoder architecture, well-suited for biomedical and materials science image segmentation.

*   **SegFormer**: A powerful and efficient segmentation model that leverages a Transformer-based encoder to capture global context, paired with a lightweight decoder.

*   **MatSegNet**: An advanced segmentation model based on a U-Net-like architecture. It features a ResNet-34 encoder and an attention mechanism in the decoder path for enhanced performance. The implementation details are as follows:

    *   **Architecture**: MatSegNet is an encoder-decoder network designed for precise semantic segmentation.

    *   **Encoder Backbone**: It uses a pre-trained **ResNet-34** model as its feature extractor. By leveraging the weights from a model trained on a large-scale dataset (like ImageNet), the encoder can extract a rich hierarchy of robust features from the input images, which is crucial for distinguishing complex structures.

    *   **Attention-Gated Decoder**: The decoder path implements a key innovation: **Attention Gates** (`AttentionBlock`). During the upsampling process, feature maps from the encoder path (via skip connections) are passed through an attention block along with the feature maps from the decoder. This mechanism teaches the model to focus on the most salient features and suppress irrelevant information from the skip connections, leading to more accurate and refined segmentation masks.

    *   **Multi-Task Learning Head**: A unique feature of this network is its **dual-output design**. The model has two prediction heads that operate in parallel from the final decoder block:
        1.  `final_mask`: Predicts the primary segmentation mask of the objects (e.g., precipitates).
        2.  `final_edge`: Predicts the boundaries or edges of these objects.
        This multi-task approach is beneficial because the edge-detection task acts as a form of regularization, forcing the model to learn more precise boundaries, which in turn improves the quality of the main segmentation mask.

    *   **Code Implementation**: The architecture is implemented in PyTorch. The `MatSegNet` class combines the ResNet-34 encoder blocks (`enc1` through `enc5`) with custom `DecoderBlock` and `AttentionBlock` modules to build the complete network. The `forward` method explicitly shows the flow of data through the encoder, attention-gated skip connections, decoder, and finally to the two output heads.

## 4. Pipeline Workflow


1.  **Data Preprocessing (Cropping & Splitting)**
    *   **Image Cropping**: Large source images are cropped into smaller patches (e.g., 512x512 pixels) for efficient model training.
    *   **Dataset Splitting**: Patches are automatically sorted into `training`, `validation`, and `test` directories with a specified ratio (e.g., 70/15/15).

2.  **Model Training**
    *   The UNet, SegFormer, and MatSegNet models are trained using the `training` and `validation` sets.
    *   Training Script: `/scripts/train.py`

3.  **Prediction & Post-processing**
    *   **Batch Prediction**: The trained model predicts segmentation masks for all image patches in the `test` set.
    *   **Result Merging**: The individual predicted masks are stitched together to reconstruct full-sized segmentation maps corresponding to the original large images.
    *   **Visualization**: Overlays of the merged predictions on the original images are generated for qualitative assessment.

4.  **Morphological Analysis**
    *   A quantitative analysis is performed on the final predicted precipitates.
    *   Analysis Script: `/scripts/carbide_morphology.py`
    *   Metrics include:
        *   Area distribution and average size
        *   Circularity or Aspect Ratio
    *   Results are saved to `/outputs/`.

## 5. Results

Performance metrics on our test set are summarized below:

    |  Model    | Accuracy | Recall   | Precision | F1-Score  |
    |  FPN      |  0.977   |  0.843   |   0.885   |   0.864   |
    |  SegFormer|  0.983   |  0.892   |   0.905   |   0.898   |
    |  U-Net    |  0.988   |  0.929   |   0.930   |   0.929   |
    |  MatSegNet|  0.989   |  0.932   |   0.936   |   0.934   |

The results clearly indicate that **MatSegNet** achieves the best performance in both Precision and Recall, highlighting its effectiveness for this segmentation task.

## 6. How to Use

1.  **Setup Environment**
    *   Clone the repository: `git clone https://github.com/Multiscale-Simulation-Lab/MatSegNet.git`
    *   Install dependencies: `pip install -r requirements.txt`
    *   Install dependencies: `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`

2.  **Prepare Data**
    *   Place your original images in the `/data/SEM_images` directory.

3.  **Run the Pipeline**
    *   Execute preprocessing: `python /scripts/python segment_images.py` and `python /scripts/python train_test_split.py`
    *   Train a model: `python /scripts/train.py --model <model_name>`
    *   Run inference and post-processing: `python /scripts/visualize_results.py  --model  <model_name>`
    *   Perform analysis: `python /scripts/carbide_morphology.py  --model  <model_name>` or `python /scripts/size_aspect_ratio.py  --model  <model_name>`

    ( <model_name> : 1.Unet, 2.Segformer, 3.MatSegNet, 4.FPN)
## 7. Project Structure

```
Segmentation_Pipeline/
├── configs/                 
│   └── config.py
│         └── MatSegNet.yaml
│         └── Segformer.yaml
│         └── Unet.yaml
├── data/                     
│   ├── SEM_images/           
│   ├── datasets/              
│      ├──bainite_set
│      ├──martensite_set
│      ├──training_set
│      ├──validation_set
│      ├──test_set
├── models/
│   ├──FPN.py
│   ├──MatSegNet.py
│   ├──Segformer.py
│   ├──Unet.py
├── output/                  
│   ├── checkpoints/
│         └──best_matsegnet.pth
│         └──best_fpn.pth
│         └──best_segformer.pth
│         └──best_unet_resnet34.pth
│         └──matsegnet.pth
│         └──fpn.pth
│         └──segformer.pth
│         └──unet_resnet34.pth
│   └── accuracy_output/     
├── src/            
│   ├── checkpoints.py
│   ├── load_data.py
│   ├── morphologies.py
│   ├── preprocessing.py
│   ├── sizes_and_aspect_ratios.py
│   ├── training.py
│   ├── transform.py
│   ├── visualization.py
├── scripts/
         └──segment_images.py
         └──train_test_split.py
         └──train.py
         └──visualize_results.py
         └──carbide_morphology.py
         └──size_aspect_ratio.py

```


### Prerequisites

numpy==1.26.4

albumentations==2.0.8

matplotlib==3.10.5

opencv-python-headless==4.10.0.84

Pillow==11.3.0

PyYAML==6.0.2

scikit-learn==1.4.2

tqdm==4.66.5

transformers==4.55.0

segmentation-models-pytorch==0.3.3


## Acknowledgements

-   This model architecture is based on the **Attention U-Net** paper: [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999).
-   The encoder implementation uses the pre-trained models provided by **PyTorch's `torchvision`**.
