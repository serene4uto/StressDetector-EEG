from torcheeg.transforms.base_transform import LabelTransform
from typing import Dict, List

class DeapAVToStress(LabelTransform):
    '''

    Args:
        threshold: list of list of float
            threshold for each level of stress in term of valence and arousal score
            (from high to low)
    '''
    def __init__(self, thresholds: List[List[float]]):
        super(DeapAVToStress, self).__init__()
        self.thresholds = thresholds

    def __call__(self, *args, y: List,
                 **kwargs) -> int:
        r'''
        Args:
            label ( list): list of labels.
            
        Returns:
            int: The output label after binarization.
        '''
        return super().__call__(*args, y=y, **kwargs)    

    def apply(self, y: List, **kwargs) -> int:
        for th_idx, th in enumerate(self.thresholds):
            if y[0] > th[0] and y[1] < th[1]:
                return len(self.thresholds) - th_idx
        return  0 # the last one is the lowest stress level
    
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'threshold': self.threshold})
