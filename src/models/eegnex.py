from torch import nn
import torch
from mmengine.model import BaseModel
from torch.nn import functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)
    
class EEGNeX(nn.Module):
    def __init__(self,
                 chunk_size: int = 128,
                 num_electrodes: int = 32,
                 F1: list = [8,32],
                 F2: list = [16,8],
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
    
        super(EEGNeX, self).__init__()

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
            nn.Conv2d(1, self.F1[0], (1, self.kernel_1), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(self.F1[0], momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.Conv2d(self.F1[0], self.F1[1], (1, self.kernel_1), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(self.F1[1], momentum=0.01, affine=True, eps=1e-3),

            Conv2dWithConstraint(self.F1[1],
                     self.F1[1] * self.D, (self.num_electrodes, 1),
                     max_norm=1,
                     stride=1,
                     padding='same',
                     groups=self.F1[1],
                     bias=False), nn.BatchNorm2d(self.F1[1] * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), 
            nn.AvgPool2d((1, 4), stride=4), 
            nn.Dropout(p=dropout)

        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                self.F1[1] * self.D,
                self.F2[0], (1, self.kernel_2),
                stride=1,
                padding="same",
                bias=False,
                groups=1,
                dilation=(1, 2)),
            nn.BatchNorm2d(self.F2[0], momentum=0.01, affine=True, eps=1e-3),

            nn.Conv2d(
                self.F2[0],
                self.F2[1], (1, self.kernel_2),
                stride=1,
                padding="same",
                bias=False,
                groups=1,
                dilation=(1, 4)),
            nn.BatchNorm2d(self.F2[1], momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=8),
            nn.Dropout(p=dropout),
            nn.Flatten(),
        )
        
        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[1]

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Forward pass of the EEGNet model.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.lin(x)

        return x
    
class MMEEGNeX(BaseModel):
    def __init__(self,
                chunk_size=128,
                num_electrodes=32,
                dropout=0.5,
                F1=[8,32],
                F2=[16,8],
                D=2,
                num_classes=2,
                kernel_1=64,
                kernel_2=16):
        
        super().__init__()

        self.eegnex = EEGNeX(
            chunk_size=chunk_size,
            num_electrodes=num_electrodes,
            dropout=dropout,
            F1=F1,
            F2=F2,
            D=D,
            num_classes=num_classes,
            kernel_1=kernel_1,
            kernel_2=kernel_2
        ).double()
    
    def forward(self, eeg_epochs, labels, mode):
        outputs = self.eegnex(eeg_epochs)
        outputs = nn.Sigmoid()(outputs) 
        
        if mode == 'loss':
            loss = F.cross_entropy(outputs, labels)
            # Also compute accuracy for training monitoring
            accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
            return {'loss': loss, 'train_accuracy': accuracy}
        elif mode == 'predict':
            return outputs, labels
