import torch
import torch.nn as nn
from torchvision import models
import os
import yaml

class UpsamplingBlock(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, out_channels, drop_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, up_in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(up_in_channels + skip_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, expansive_input, contractive_input):
        x = self.up(expansive_input)
        # Pad if shapes don't match
        if x.size()[2:] != contractive_input.size()[2:]:
            diffY = contractive_input.size()[2] - x.size()[2]
            diffX = contractive_input.size()[3] - x.size()[3]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, contractive_input], dim=1)
        x = self.conv(x)
        return x


class UnetResNet34(nn.Module):
    def __init__(self, n_classes=1, dropout_rates=None):
        super().__init__()

        # Default dropout rates: bottleneck, d1, d2, d3, d4
        if dropout_rates is None:
            dropout_rates = [0.4, 0.4, 0.3, 0.2, 0.1]

        base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.encoder = nn.ModuleDict({
            'conv1': nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu),
            'maxpool': base_model.maxpool,
            'layer1': base_model.layer1,
            'layer2': base_model.layer2,
            'layer3': base_model.layer3,
            'layer4': base_model.layer4
        })

        self.bottleneck_dropout = nn.Dropout2d(p=dropout_rates[0])

        # Decoder with decaying dropout
        self.d1 = UpsamplingBlock(512, 256, 256, drop_p=dropout_rates[1])
        self.d2 = UpsamplingBlock(256, 128, 128, drop_p=dropout_rates[2])
        self.d3 = UpsamplingBlock(128, 64, 64, drop_p=dropout_rates[3])
        self.d4 = UpsamplingBlock(64, 64, 64, drop_p=dropout_rates[4])

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        s1 = self.encoder.conv1(x)
        p1 = self.encoder.maxpool(s1)
        s2 = self.encoder.layer1(p1)
        s3 = self.encoder.layer2(s2)
        s4 = self.encoder.layer3(s3)
        b = self.encoder.layer4(s4)
        b = self.bottleneck_dropout(b)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        final = self.final_up(d4)
        outputs = self.outputs(final)
        return outputs


def get_model(n_classes=None, dropout_rates=[0.4, 0.4, 0.3, 0.2, 0.1],
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    if n_classes is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        configs_dir = os.path.abspath(os.path.join(project_root, 'configs'))
        CONFIG_FILE_PATH = os.path.join(configs_dir, "Unet.yaml")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = yaml.safe_load(f)
        n_classes = config['num_classes']

    model = UnetResNet34(n_classes=n_classes, dropout_rates=dropout_rates)
    model = model.to(device)
    return model


if __name__ == "__main__":
    model_default = get_model()
    model_custom = get_model(n_classes=5)
    print("Model loaded successfully!")
