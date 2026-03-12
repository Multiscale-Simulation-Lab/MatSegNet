import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import yaml


class FPNEfficientNetB4(nn.Module):
    def __init__(self, n_classes=1, dropout_rate=0.3):
        super().__init__()
        base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        encoder_features = base_model.features

 
        self.encoder = nn.ModuleDict({
            'stage1_2': nn.Sequential(*encoder_features[0:3]), 
            'stage3':   encoder_features[3],                  
            'stage4':   nn.Sequential(*encoder_features[4:6]), 
            'stage5':   nn.Sequential(*encoder_features[6:9])  
        })

        # --- 3. Define FPN decoder (lateral, smooth layers) ---
        fpn_out_channels = 256
        
        self.lateral_c5 = nn.Conv2d(1792, fpn_out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(160, fpn_out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(56, fpn_out_channels, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(32, fpn_out_channels, kernel_size=1)
        
        self.smooth_p4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        
        # --- 4. Define Segmentation Head ---
        self.dropout = nn.Dropout2d(p=dropout_rate)
        head_in_channels = fpn_out_channels * 4
        head_out_channels = 256
        
        self.conv_head = nn.Sequential(
            nn.Conv2d(head_in_channels, head_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_out_channels),
            nn.ReLU(inplace=True)
        )
        self.outputs = nn.Conv2d(head_out_channels, n_classes, kernel_size=1)

    def _upsample_add(self, p, c):
        p_upsampled = F.interpolate(p, size=c.shape[2:], mode='bilinear', align_corners=False)
        return p_upsampled + c

    def forward(self, x):
       
        c2_out = self.encoder.stage1_2(x)
  
        c3_out = self.encoder.stage3(c2_out)
 
        c4_out = self.encoder.stage4(c3_out)
   
        c5_out = self.encoder.stage5(c4_out)
  
        
        p5 = self.lateral_c5(c5_out)
 
        p4 = self._upsample_add(p5, self.lateral_c4(c4_out))
 
        p4 = self.smooth_p4(p4)
 
        p3 = self._upsample_add(p4, self.lateral_c3(c3_out))
 
        p3 = self.smooth_p3(p3)
 
        p2 = self._upsample_add(p3, self.lateral_c2(c2_out))
 
        p2 = self.smooth_p2(p2)

        
     
        p5_up = F.interpolate(p5, size=p2.shape[2:], mode='bilinear', align_corners=False)

        
        p4_up = F.interpolate(p4, size=p2.shape[2:], mode='bilinear', align_corners=False)

        
        p3_up = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False)

        
        p_concat = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)

        
        p_concat = self.dropout(p_concat)

        
        head_out = self.conv_head(p_concat)

        
        final_up = F.interpolate(head_out, scale_factor=4, mode='bilinear', align_corners=False)

                
        outputs = self.outputs(final_up)

        return outputs


def get_model(n_classes=None, dropout_rate=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    if n_classes is None or dropout_rate is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        configs_dir = os.path.abspath(os.path.join(project_root, 'configs'))
        CONFIG_FILE_PATH = os.path.join(configs_dir, "FPN.yaml")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = yaml.safe_load(f)

        if n_classes is None:
            n_classes = config['num_classes']
        
        if dropout_rate is None:
            dropout_rate = config.get('train', {}).get('dropout_rate', 0.3) 

    model = FPNEfficientNetB4(n_classes=n_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    return model


 

if __name__ == "__main__":
    model_default = get_model(n_classes=None, dropout_rate=None)
    model_custom = get_model(n_classes=5, dropout_rate=0.5)
    print("model has been constructed")
  