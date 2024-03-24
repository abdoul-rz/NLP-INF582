import torch
from torch.autograd import Variable

class NERDataset(torch.utils.data.Dataset):

  def __init__(self, vecs, labels):
    self.labels = labels
    self.vecs = vecs

  def __len__(self):
    return len(self.vecs)

  def __getitem__(self, index):
    batch_data=self.vecs[index]
    batch_labels=self.labels[index]

    batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
    #convert Tensors to Variables
    #batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

    return batch_data, batch_labels
