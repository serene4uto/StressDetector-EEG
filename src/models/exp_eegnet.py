from torch import nn
import torch
from torch.nn import functional as F
from mmengine.model import BaseModel

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

# class ResidualBlock1D(nn.Module):
#     def __init__(self, in_channels: int, 
#                  out_channels: int, 
#                  kernel_size: int, 
#                  stride: int = 1, 
#                  padding: int = 0, 
#                  dilation: int = 1, 
#                  groups: int = 1, 
#                  bias: bool = True, 
#                  padding_mode: str = 'zeros', 
#                  dropout: float = 0.25):
#         super(ResidualBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout(p=dropout)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#         self.bn2 = nn.BatchNorm1d(out_channels)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.elu(out)
#         out = self.dropout(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity
#         out = self.elu(out)

#         return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResEEGNet(nn.Module):
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(ResEEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), 
            nn.AvgPool2d((1, 4), stride=4), 
            nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), 
            nn.ELU(), 
            # nn.AvgPool2d((1, 8), stride=8),
            # nn.Dropout(p=dropout)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
            ),
            ResidualBlock(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
            ),
            nn.AvgPool2d((1, 8), stride=8),
            # nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)

            # print(mock_eeg.shape)

        return self.F2 * mock_eeg.shape[3]
        # return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x
    
class MMResEEGNet(BaseModel):
    def __init__(self,
                chunk_size=128,
                num_electrodes=32,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=8,
                F2=16,
                D=2,
                num_classes=2):
        
        super().__init__()
        self.reseegnet = ResEEGNet(
            chunk_size=chunk_size,
            num_electrodes=num_electrodes,
            dropout=dropout,
            kernel_1=kernel_1,
            kernel_2=kernel_2,
            F1=F1,
            F2=F2,
            D=D,
            num_classes=num_classes
        ).double()
        
    def forward(self, eeg_epochs, labels, mode):
        outputs = self.reseegnet(eeg_epochs.unsqueeze(1))
        outputs = nn.Sigmoid()(outputs) 
        
        if mode == 'loss':
            return {'loss': F.cross_entropy(outputs, labels)}
        elif mode == 'predict':
            return outputs, labels