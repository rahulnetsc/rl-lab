import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, img_height:int, img_width:int, ) -> None:
        super().__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,out_channels=32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_height, img_width)
            dummy_output = self._feature_extractor(dummy_input)
            flatten_dim = dummy_output.view(1,-1).shape[1]
        
        self._layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_channels)
        )

    def forward(self,x):
        assert x.shape[1] == self.in_channels,  (
            f"Input channels: {x.shape[1]} does not match CNN in_channels: {self.in_channels}"
        )
        x = self._feature_extractor(x)
        x = self._layers(x)
        return x
    