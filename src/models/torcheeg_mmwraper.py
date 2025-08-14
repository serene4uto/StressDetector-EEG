from mmengine.model import BaseModel
from torcheeg.models import EEGNet, FBCNet, TSCeption
import torch.nn.functional as F
from torch import nn

class MMEEGNet(BaseModel):
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
        self.eegnet = EEGNet(
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
        outputs = self.eegnet(eeg_epochs)
        outputs = nn.Sigmoid()(outputs) 
        
        if mode == 'loss':
            loss = F.cross_entropy(outputs, labels)
            # Also compute accuracy for training monitoring
            accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
            return {'loss': loss, 'train_accuracy': accuracy}
        elif mode == 'predict':
            return outputs, labels


class MMFBCNet(BaseModel):
    def __init__(self,
                 num_classes=2,
                 num_electrodes=32,
                 chunk_size=128,
                 in_channels=1,
                 num_S=32):
        super().__init__()

        self.fbcnet = FBCNet(
            num_classes=num_classes,
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            in_channels=in_channels,
            num_S=num_S
        ).double()
        
    def forward(self, eeg_epochs, labels, mode):
        outputs = self.fbcnet(eeg_epochs)
        if mode == 'loss':
            loss = F.cross_entropy(outputs, labels)
            # Also compute accuracy for training monitoring
            accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
            return {'loss': loss, 'train_accuracy': accuracy}
        elif mode == 'predict':
            return outputs, labels
        
class MMTSCeption(BaseModel):
    def __init__(self,
                 num_classes=2,
                 num_electrodes=32,
                 sampling_rate=128,
                 num_T=15,
                 num_S=15,
                 hid_channels=32,
                 dropout=0.5):
        super().__init__()

        self.tsception = TSCeption(
            num_classes=num_classes,
            num_electrodes=num_electrodes,
            sampling_rate=sampling_rate,
            num_T=num_T,
            num_S=num_S,
            hid_channels=hid_channels,
            dropout=dropout
        ).double()
        
    def forward(self, eeg_epochs, labels, mode):
        outputs = self.tsception(eeg_epochs)
        if mode == 'loss':
            loss = F.cross_entropy(outputs, labels)
            # Also compute accuracy for training monitoring
            accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
            return {'loss': loss, 'train_accuracy': accuracy}
        elif mode == 'predict':
            return outputs, labels