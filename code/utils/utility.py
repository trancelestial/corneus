import numpy as np
import shutil
import os
import torch
from torch.utils.data import DataLoader

def generate_data_folders(base_path, typ, img_paths, labels, split=None):
    # split - percentage of validation data
    l_unique = labels.unique()
    for l in l_unique:
        try:
            os.makedirs(os.path.join(base_path, typ, l))
            if split is not None:
                os.makedirs(os.path.join(base_path, 'val', l))
        except FileExistsError:
            # directory already exists
            pass

    for i in range(len(img_paths)):
        if (split is not None) and (np.random.rand() <= split):
            wr_path = os.path.join(base_path, 'val')
        else:
            wr_path = os.path.join(base_path, typ)

        shutil.move(os.path.join(base_path, typ, img_paths[i]), os.path.join(wr_path, labels[i], img_paths[i]))

def calculate_normalization(inspect_data, batch_size=128, shuffle=False):
    imgs = []
    inspect_loader = DataLoader(inspect_data, batch_size=128, shuffle=True)
    for img, _ in inspect_loader:
        imgs.append(img)
    ti = torch.cat(imgs, axis=0)
    
    return torch.mean(ti, dim=(0,2,3)), torch.std(ti, dim=(0,2,3)) 
