from .SAM40 import make_SAM40_dataset

make_dataset_function = {
    'SAM40' : make_SAM40_dataset.make_dataset
}

def make_dataset(dataset_name: str = None):
    if dataset_name is None:
        raise ValueError('dataset_name is None')
    if dataset_name not in make_dataset_function:
        raise ValueError('dataset_name is not in make_dataset_function')
    
    return make_dataset_function[dataset_name]

class DatasetMaker():
    def __init__(self, dataset_name: str = None):
        self.make_dataset = make_dataset(dataset_name)
    
    
