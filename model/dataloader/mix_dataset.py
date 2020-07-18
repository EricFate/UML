from torch.utils.data import Dataset


class MixedDatasetWrapper(Dataset):
    def __init__(self, sd, ud):
        '''

        :param sd: supervised dataloader
        :param ud: unsupervised dataloader
        '''

        self.sd = sd
        self.ud = ud
        self.s_iter = iter(sd)
        self.u_iter = iter(ud)
        self.count = 0

    def __getitem__(self, index):
        s_image, _ = next(self.s_iter)
        u_image, _ = next(self.u_iter)
        self.count += 1
        if self.count >= self.__len__():
            self.s_iter = iter(self.sd)
            self.u_iter = iter(self.ud)
            self.count = 0
        return s_image, u_image

    def __len__(self):
        return min(len(self.sd), len(self.ud))
