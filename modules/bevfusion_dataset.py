from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch

import random
import threading

log_lock = threading.Lock() 

def log_missing_file(path, log_file='missing_files.txt'):
    with log_lock:
        with open(log_file, 'a') as f:
            f.write(path + '\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BevfusionDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform
        self.bevfusion_dir = '/data/Datasets/NuScenes-QA/bevfusion_feats/DriveLM/all'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Get the question and answer at the idx
            qa, img_path = self.data[idx]
            img_path = list(img_path.values())

            bevfusion_path = os.path.join(self.bevfusion_dir, os.path.splitext(os.path.basename(img_path[3]))[0] + '.pt')

            if not os.path.exists(bevfusion_path):
                # log_missing_file(bevfusion_path)
                raise FileNotFoundError

                
            bevfusion_feat = torch.load(bevfusion_path)

            q_text, a_text = qa['Q'], qa['A']
            q_text = f"Question: {q_text} Answer:"

            # # Concatenate images into a single tensor
            # imgs = [self.transform(read_image(p).float()).to(device) for p in img_path]
            # imgs = torch.stack(imgs, dim=0)

            return q_text, bevfusion_feat, a_text, sorted(list(img_path))

        except (FileNotFoundError, OSError):
            return self.__getitem__(random.randint(0, len(self.data) - 1))

    def collate_fn(self, batch):
        q_texts, bevfusion_feat, a_texts, _ = zip(*batch)
        bevfusion_feat = torch.stack(list(bevfusion_feat), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids

        return encodings, bevfusion_feat, labels

    def test_collate_fn(self, batch):
        q_texts, bevfusion_feat, a_texts, img_path = zip(*batch)
        bevfusion_feat = torch.stack(list(bevfusion_feat), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids

        return list(q_texts), encodings, bevfusion_feat, labels, img_path
