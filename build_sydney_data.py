# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset, DataLoader

import glob
import time
import random


def read_bin(file_name):
    """
    This script is provided by data creator
    http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml
    """
    names = ['t','intensity','id','x','y','z','azimuth','range','pid']

    formats = ['int64', 'uint8', 'uint8',
               'float32', 'float32', 'float32',
               'float32', 'float32', 'int32']

    binType = np.dtype( dict(names=names, formats=formats) )
    data = np.fromfile(file_name, binType)

    # 3D points, one per row
    P = np.vstack([ data['x'], data['y'], data['z'] ]).T
    
    return P

def labels():
    labels_dict = {'4wd': 0,
                  'bench': 1,
                  'bicycle': 2,
                  'biker': 3,
                  'building': 4,
                  'bus': 5,
                  'car': 6,
                  'cyclist': 7,
                  'excavator': 8,
                  'pedestrian': 9,
                  'pillar': 10,
                  'pole': 11,
                  'post': 12,
                  'scooter': 13,
                  'ticket_machine': 14,
                  'traffic_lights': 15,
                  'traffic_sign': 16,
                  'trailer': 17,
                  'trash': 18,
                  'tree': 19,
                  'truck': 20,
                  'trunk': 21,
                  'umbrella': 22,
                  'ute': 23,
                  'van': 24,
                  'vegetation': 25}

    num_classes = len(labels_dict.keys())
    return num_classes, labels_dict


def sub_sample(pcs, labels, k):    
    res1 = np.concatenate((pcs, np.reshape(labels, (labels.shape[0], 1))), axis= 1)
    res = np.asarray(random.choices(res1, weights=None, cum_weights=None, k=k))
    pcs = res[:, 0:-1]
    labels = res[:, -1]
    labels -= 1
    return pcs, labels


class SydneyUrban(Dataset):

    def __init__(self, transform=False):
        self.files = []
        self.labels = []
        self.sample_size =  50
        
        folders = glob.glob("./sydney-urban-objects-dataset/objects")
        files = glob.glob(folders[0] + "/*")
        for file in files:
            file_name = file.split("/")[-1]
            if file_name.split(".")[-1] == "bin":
                label = file_name.split(".")[0]
                self.labels.append(labels_dict[label])
                self.files.append(file) 
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts_path = self.files[idx]
        label = self.labels[idx]
        
        pts = read_bin(pts_path)
        label = [label]*len(pts)
        label = np.array(label)
        
        pts, label = sub_sample(pts, label, self.sample_size)
        label = label[0]+1
        #label = np.expand_dims(label, 1)
        return {'points': np.array(pts, dtype="float32"), 'labels': label.astype(int)}

#sample = read_bin('./sydney-urban-objects-dataset/objects/4wd.1.2446.bin')
