import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):

    def __init__(self, in_c, out_c, drop_p=0.3):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if drop_p and drop_p > 0:
            layers.append(nn.Dropout(drop_p))
        layers += [
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class MatSegNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)


        self.encoder = nn.ModuleDict({
            'enc1': nn.Sequential(*list(base.children())[:3]),   # conv1 + bn1 + relu -> out 64, /2
            'enc2': nn.Sequential(*list(base.children())[3:5]),  # maxpool + layer1 -> out 64, /4
            'enc3': list(base.children())[5],  # layer2 -> out 128, /8
            'enc4': list(base.children())[6],  # layer3 -> out 256, /16
            'enc5': list(base.children())[7]   # layer4 -> out 512, /32
        })


        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

  
        self.att5 = AttentionBlock(F_g=256, F_l=256, F_int=128)  
        self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)   
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)  
        self.att2 = AttentionBlock(F_g=32, F_l=64, F_int=32)  

        # Decoder blocks: in_channels = concat(attended_skip, up_feature)
        self.dec5 = DecoderBlock(in_c=256 + 256, out_c=256, drop_p=0.4)  # concat(e4:256, d5:256)
        self.dec4 = DecoderBlock(in_c=128 + 128, out_c=128, drop_p=0.3)  # concat(e3:128, d4:128)
        self.dec3 = DecoderBlock(in_c=64 + 64, out_c=64, drop_p=0.2)     # concat(e2:64, d3:64)
        self.dec2 = DecoderBlock(in_c=32 + 64, out_c=32, drop_p=0.1)     # concat(e1:64, d2:32) -> 96 before conv, but we set in_c=96? careful below

        # Note: dec2 expects 96 in channels (32 from up, 64 from skip). To keep type safe we will build it using the correct in_c:
        # Recreate dec2 with correct in_c:
        self.dec2 = DecoderBlock(in_c=32 + 64, out_c=32, drop_p=0.1)

        # Final up + refinement
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = DecoderBlock(16, 16, drop_p=0.0)

        # Output heads
        self.final_mask = nn.Conv2d(16, n_classes, kernel_size=1)
        self.final_edge = nn.Conv2d(16, 1, kernel_size=1)

    def _match_size(self, src, dst):
        """
        Ensure src has same spatial size as dst by interpolating if needed.
        Uses bilinear interpolation for feature maps.
        """
        if src.size()[2:] == dst.size()[2:]:
            return src
        return F.interpolate(src, size=dst.size()[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        e1 = self.encoder.enc1(x)  # /2
        e2 = self.encoder.enc2(e1) # /4
        e3 = self.encoder.enc3(e2) # /8
        e4 = self.encoder.enc4(e3) # /16
        e5 = self.encoder.enc5(e4) # /32

        # Decoder level 5 (bottom -> up)
        d5 = self.up5(e5)                   # expected out channels 256
        d5 = self._match_size(d5, e4)       # ensure size matches enc4
        a4 = self.att5(g=d5, x=e4)          # attention on enc4
        d5 = torch.cat([a4, d5], dim=1)     # concat -> 512 channels
        d5 = self.dec5(d5)                  # -> 256 channels

        # Decoder level 4
        d4 = self.up4(d5)                   # -> 128
        d4 = self._match_size(d4, e3)
        a3 = self.att4(g=d4, x=e3)
        d4 = torch.cat([a3, d4], dim=1)     # -> 256
        d4 = self.dec4(d4)                  # -> 128

        # Decoder level 3
        d3 = self.up3(d4)                   # -> 64
        d3 = self._match_size(d3, e2)
        a2 = self.att3(g=d3, x=e2)
        d3 = torch.cat([a2, d3], dim=1)     # -> 128
        d3 = self.dec3(d3)                  # -> 64

        # Decoder level 2 (final skip from enc1)
        d2 = self.up2(d3)                   # -> 32
        d2 = self._match_size(d2, e1)
        a1 = self.att2(g=d2, x=e1)
        d2 = torch.cat([a1, d2], dim=1)     # -> 96 (64 + 32)
        d2 = self.dec2(d2)                  # -> 32

        # Final up to input resolution (note: e1 is /2, so final_up restores to original)
        final = self.final_up(d2)           # -> 16
        final = self._match_size(final, x)  # ensure exact match to input size
        final = self.final_conv(final)      # refine -> 16

        mask = self.final_mask(final)
        edge = self.final_edge(final)
        return mask, edge



def get_model(n_classes=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load model. If n_classes is None, try reading a config file ../configs/MatSegNet.yaml
    Otherwise default n_classes=1.
    """
    if n_classes is None:
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            configs_dir = os.path.abspath(os.path.join(project_root, 'configs'))
            CONFIG_FILE_PATH = os.path.join(configs_dir, "MatSegNet.yaml")
            with open(CONFIG_FILE_PATH, 'r') as f:
                config = yaml.safe_load(f)
            n_classes = int(config.get('num_classes', 1))
        except Exception as e:
            print(f"Warning: could not load config file ({e}), using default n_classes=1")
            n_classes = 1

    model = MatSegNet(n_classes=n_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # quick sanity check with random tensors (cpu)
    device = torch.device("cpu")
    model = get_model(n_classes=1, device=device)
    model.eval()
    # test with a few different sizes to ensure _match_size works
    for H, W in [(256,256), (300,300), (512,384)]:
        x = torch.randn(1, 3, H, W, device=device)
        with torch.no_grad():
            mask, edge = model(x)
        print(f"Input {H}x{W} -> mask {mask.shape}, edge {edge.shape}")
    print("Model loaded and basic forward checks passed.")
