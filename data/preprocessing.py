import numpy as np
import h5py
import torch
import random
from torchvision.transforms import transforms

def load(data_path):
    image_path = data_path[0]

    if 'hdf5' in image_path:
        f = h5py.File(image_path, 'r')
        image_numpy = np.array(f.get('image'), dtype = np.float32)
    return image_numpy

class Random_select(object):
    def __init__(self, mode='train', ixi=False):
        self.mode = mode
        self.ixi = ixi
    def __call__(self, sample):
        image = sample['image']       
        if self.mode == 'test':
                return {'image': image}

        image_masked = np.copy(image)
        if self.ixi:
            modalitiy = random.randint(0,2)
        else:
            modalitiy = random.randint(0,3)
        image_masked[modalitiy,...] = -1 #zero value but we normalized to -1~1
        target = image[modalitiy:modalitiy+1,...]
        return {'image': image,'image_masked': image_masked, 'target': target, 'modalitiy': modalitiy}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mode='train', ixi=False):
        self.mode = mode
        self.ixi = ixi
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float()

        if self.mode == 'test':
            return {'image': image}

        target = sample['target']
        target = np.ascontiguousarray(target)
        target = torch.from_numpy(target).float()

        modalitiy = sample['modalitiy']
        modalitiy = np.ascontiguousarray(modalitiy)
        modalitiy = torch.from_numpy(modalitiy).long()
        
        image_masked = sample['image_masked']
        image_masked = np.ascontiguousarray(image_masked)
        image_masked = torch.from_numpy(image_masked).float()

        return {'image': image,'image_masked': image_masked, 'target': target, 'modalitiy': modalitiy}
    
def augmentation(sample, ixi=False):
    trans = transforms.Compose([
        Random_select(ixi=ixi),
        ToTensor(ixi=ixi)
    ])

    return trans(sample)

def augmentation_test(sample, ixi=False):
    trans = transforms.Compose([
        Random_select(mode='test', ixi=ixi),
        ToTensor(mode='test', ixi=ixi)
    ])

    return trans(sample)