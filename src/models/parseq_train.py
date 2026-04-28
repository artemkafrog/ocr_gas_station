import sys
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
from torchvision import transforms
os.chdir('')
print("Current working directory:", os.getcwd())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parseq'))

class GasDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['file']
        if img_path.startswith('../'):
            img_path = img_path[3:]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, item['text']


class GasDataModule(pl.LightningDataModule):
    def __init__(self, train_json, val_json, batch_size=8):
        super().__init__()
        self.train_json = train_json
        self.val_json = val_json
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = GasDataset(self.train_json)
        self.val_dataset = GasDataset(self.val_json)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

from parseq.strhub.models.parseq.system import PARSeq
from parseq.strhub.data.utils import Tokenizer

charset = '0123456789.'
tokenizer = Tokenizer(charset=charset)

model = PARSeq(
    num_tokens=len(tokenizer),
    max_label_length=8,
    img_size=(32, 128),
    patch_size=(4, 8),
    embed_dim=384,
    enc_num_heads=6,
    enc_mlp_ratio=4,
    enc_depth=12,
    dec_num_heads=12,
    dec_mlp_ratio=4,
    dec_depth=1,
    decode_ar=True,
    refine_iters=1,
    dropout=0.1,

    charset_train=charset,
    charset_test=charset,
    batch_size=8,
    lr=7e-4,
    warmup_pct=0.2,
    weight_decay=0.1,
    perm_num=6,
    perm_forward=True,
    perm_mirrored=True
)

data_module = GasDataModule(
    train_json='data/prepared/annotations/raw_train.json',
    val_json='data/prepared/annotations/raw_val.json',
    batch_size=8
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='cpu',
    devices=1
)

print("\nStarting learning PARSeq...")
trainer.fit(model, data_module)

trainer.save_checkpoint('parseq_gas_model.ckpt')
print("\nModel PARSeq saved as parseq_gas_model.ckpt")