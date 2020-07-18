import torch
import numpy as np


class DummyWrapper(object):
    def __init__(self, num_dummy, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.range = np.arange(len(self.dataset))
        self.num_dummy = num_dummy

    def dummy_loader(self, loader):
        for input, target in loader:
            b_unsupervised = self.dataset.unsupervised
            self.dataset.unsupervised = False
            idx = np.random.choice(self.range, self.num_dummy, replace=False)
            dummy = [self.dataset[i][0] for i in idx]
            dummy = torch.stack(dummy)
            self.dataset.unsupervised = b_unsupervised
            yield input, dummy

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.dummy_loader(self.dataloader)
