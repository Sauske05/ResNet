import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
# test_files = ['test_batch']


# byte_dict = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')

# data_dict = {key.decode('utf-8') if isinstance(key, bytes) else key:value for key, value in byte_dict.items()}
class  CustomDataset(Dataset):
    def __init__(self, files: List[str]):
        super().__init__()
        self.files = files
        self.images = []
        self.labels = []

        for file in self.files:
            byte_dict = unpickle(f'./cifar-10-python/cifar-10-batches-py/{file}')
            data_dict = {key.decode('utf-8') if isinstance(key, bytes) else key:value for key, value in byte_dict.items()}
            images = data_dict['data'].reshape(-1,3,32,32)
            #images = torch.tensor(images, dtype = torch.float32)
            images = torch.tensor(images, dtype=torch.float32) / 255.0
            labels = data_dict['labels']
            labels = torch.tensor(labels)


            self.images.append(images)
            self.labels.append(labels)
        
        self.images = torch.vstack(self.images)
        self.labels = torch.hstack(self.labels)
    # def transform(self):
    #     for file in self.files:
    #         byte_dict = unpickle(f'./cifar-10-python/cifar-10-batches-py/{file}')
    #         data_dict = {key.decode('utf-8') if isinstance(key, bytes) else key:value for key, value in byte_dict.items()}
    #         for 
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
        
    def __len__(self):
        return len(self.images)
    


# train_dataset = CustomDataset(train_files)
# test_dataset = CustomDataset(test_files)


# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)