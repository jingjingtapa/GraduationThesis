import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import BEVDataset
class BEVDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers #os.cpu_count() 로 확인
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = BEVDataset(split='train')
            self.val_dataset = BEVDataset(split='val')
        elif stage == 'test':
            self.test_dataset = BEVDataset(split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

