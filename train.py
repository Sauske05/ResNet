import torch
from torch import nn
from model import ResNet50
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from typing import List, Tuple

from tqdm import tqdm
def load_dataset(train_files : List[str], test_files: List[str]) -> Tuple[CustomDataset]:
    train_dataset = CustomDataset(train_files)
    test_dataset = CustomDataset(test_files)
    return train_dataset, test_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_files = ['test_batch']


def load_dataloader(train_files: List[str], test_files: List[str]) -> Tuple[DataLoader]:
    train_dataset, test_dataset = load_dataset(train_files, test_files)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataloader, test_dataloader

def accuracy(y_pred, y_actual):
    #y_actual = y_actual.long()
    y_pred_argmax = torch.argmax(y_pred, dim=-1)
    return (torch.sum(y_pred_argmax == y_actual).item() / len(y_actual)) * 100

epoches:int = 50
loss:nn.CrossEntropyLoss = nn.CrossEntropyLoss()
lr:float = 1e-3
model:ResNet50 = ResNet50([3, 4, 6, 3]).to(device=device)
optimizer:torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
train_dataloader, test_dataloader = load_dataloader(train_files, test_files)

def train() -> None:
    
    for epoch in tqdm(range(epoches)):
        epoch_train_loss = 0
        epoch_val_loss = 0
        
        model.train()
        epoch_train_accuracy, epoch_val_accuracy = 0,0
        for image_batch, label_batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            y_pred = model(image_batch)
            #print(y_pred.shape)
            #print(label_batch.shape)
            batch_loss = loss(y_pred, label_batch)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss
            train_accuracy = accuracy(y_pred, label_batch)
            epoch_train_accuracy += train_accuracy
        average_train_accuracy = epoch_train_accuracy / len(train_dataloader)
        average_epoch_loss = epoch_train_loss / len(train_dataloader)
        model.eval()
        
        with torch.no_grad():
            for image_batch, label_batch in tqdm(test_dataloader):
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                y_val = model(image_batch)
                batch_loss = loss(y_val, label_batch)
                epoch_val_loss +=batch_loss
                val_accuracy = accuracy(y_val, label_batch)
                epoch_val_accuracy += val_accuracy
        average_val_accuracy = epoch_val_accuracy / len(test_dataloader)
        average_val_loss = epoch_val_loss / len(test_dataloader)
        
        print(f'Average Train Loss in Epoch {epoch+1} --> {average_epoch_loss} & Average Val Loss in Epoch {epoch+1} --> {average_val_loss}')
        print(f'Average Train Accuracy in Epoch {epoch+1} --> {average_train_accuracy} & Average Val Accuracy in Epoch {epoch+1} --> {average_val_accuracy}')
                
        





# train_dataloader, test_dataloader = load_dataloader(train_files, test_files)

# for image_batch, label_batch in train_dataloader:
#     # print(index)
#     # print(batch)
#     print(type(train_dataloader))
#     try:
#         print(image_batch.shape)
#         print(label_batch.shape)
#     except Exception as e:
#         print(e)
#     break



def test():
    resnet50 = ResNet50([3, 4, 6, 3])  
    image_sample = torch.rand(2,3,32,32)
    output = resnet50(image_sample)
    output_argmax = torch.argmax(output, dim=1)
    labels = torch.tensor([1,2])
    print(output_argmax)
    print(labels)

    print(f'Accuracy : {accuracy(output, labels)}')


if __name__ == '__main__':
    train()