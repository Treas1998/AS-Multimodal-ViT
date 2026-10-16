import torch
import torch.nn as nn
import timm


class MultiModalViT_timm(nn.Module):
    def __init__(self, num_tabular_features=5, local_vit_weight_path=None):
        super().__init__()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)


        if local_vit_weight_path is not None:
            state_dict = torch.load(local_vit_weight_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded local vit weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")


        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )


        self.classifier = nn.Sequential(
            nn.Linear(self.vit.num_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, image, tabular):
        img_feat = self.vit(image)
        tab_feat = self.tabular_net(tabular)
        combined = torch.cat((img_feat, tab_feat), dim=1)
        return self.classifier(combined)
