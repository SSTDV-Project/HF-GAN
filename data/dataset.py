import os
from torch.utils.data import Dataset
from data.preprocessing import load, augmentation, augmentation_test

class BraTSDataset(Dataset):
    """BraTS Dataset""" 
    def __init__(self,
                 dataset_path,
                 mode='train',
                 ):

        self.mode = mode
        
        self.dataset_path = os.path.join(dataset_path, mode)
        img_names = [d for d in os.listdir(self.dataset_path) if d.endswith('hdf5')]
        img_names.sort()
        self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        image = load(self.data_path[idx])
        sample = {'image':image}
        if self.mode == 'test':
            output = augmentation_test(sample)
        else:
            output = augmentation(sample)  
        return output


class IXIDataset(Dataset):
    """IXI Dataset""" 
    def __init__(self,
                 dataset_path,
                 mode='train',
                 ):

        self.mode = mode
        
        self.dataset_path = os.path.join(dataset_path, mode)
        img_names = [d for d in os.listdir(self.dataset_path) if d.endswith('hdf5')]
        self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        image = load(self.data_path[idx])
        sample = {'image':image}
        if 'test' in self.mode:
            output = augmentation_test(sample, ixi=True)
        else:
            output = augmentation(sample, ixi=True)
        return output
