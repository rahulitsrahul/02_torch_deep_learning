import torch
from generate_dataset import *


class dataloader_handler(object):
    def __init__(self, df, unique_imgs, train_inds, train_imgs_folder, im_ext, batch_size, shuffle):
        self.df = df
        self.unique_imgs = unique_imgs
        self.train_inds = train_inds
        self.train_imgs_folder = train_imgs_folder
        self.im_ext = im_ext
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataloader_obj = None
        
        self.get_DataLoader()
        
    def custom_collate(self, data):
        return data
    
    def get_DataLoader(self):
        self.dataloader_obj = torch.utils.data.DataLoader(
                                    generate_dataset(self.df, self.unique_imgs, self.train_inds, self.train_imgs_folder, self.im_ext),
                                                batch_size=self.batch_size,
                                                shuffle=self.shuffle,
                                                collate_fn = self.custom_collate,
                                                pin_memory = True if torch.cuda.is_available() else False
                                                )
        
        