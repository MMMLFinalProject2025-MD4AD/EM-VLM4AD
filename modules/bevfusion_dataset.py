from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BevfusionDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform
        self.bevfusion_dir = '/data/Datasets/NuScenes-QA/bevfusion_feats/DriveLM/train'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        qa, img_path = self.data[idx]
        img_path = list(img_path.values())

        bevfusion_path = os.path.join(self.bevfusion_dir, os.path.splitext(os.path.basename(img_path[3]))[0] + '.pt')
        bevfusion_feat = torch.load(bevfusion_path).to(device)

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # # Concatenate images into a single tensor
        # imgs = [self.transform(read_image(p).float()).to(device) for p in img_path]
        # imgs = torch.stack(imgs, dim=0)

        return q_text, bevfusion_feat, a_text, sorted(list(img_path))

    def collate_fn(self, batch):
        q_texts, bevfusion_feat, a_texts, _ = zip(*batch)
        bevfusion_feat = torch.stack(list(bevfusion_feat), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return encodings, bevfusion_feat, labels

    def test_collate_fn(self, batch):
        q_texts, bevfusion_feat, a_texts, img_path = zip(*batch)
        bevfusion_feat = torch.stack(list(bevfusion_feat), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return list(q_texts), encodings, bevfusion_feat, labels, img_path
