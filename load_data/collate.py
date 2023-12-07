import torch
import numpy as np
import librosa
import sklearn
from sklearn import preprocessing
from load_data.custom_dataset import CustomDataset


def process_data(np_wav):
    # Image augmentation
    transforms = v2.Compose([
        v2.Grayscale(1),
        v2.RandomResizedCrop(size=(1024, 1024), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.tensor(np.expand_dims(np.swapaxes(pad,0,1), axis=0), dtype=torch.float)


def collate_fn(batch):
    features_batch, labels_batch = zip(*batch)
    
    processed_batch = []
    for data in features_batch:
        #processed_data = process_data(data)
        processed_data = torch.tensor(data, dtype=torch.float)
        processed_batch.append(processed_data)
    return torch.stack(processed_batch), torch.tensor(labels_batch)