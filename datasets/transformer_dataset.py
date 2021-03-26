import torch
import torch.nn as nn

class TransformersDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, len):
      """
        Args:
            dataset: token dataset
            ttps: tokens per second
            seconds: seconds per example
      """
      
      self.ids = []
      self.annotations = []
      self.dataset = []
     
      for k, data in enumerate(dataset):
        for i in range(0, len(data)-len, len):
          self.ids.append(dataset['ids'][k])
          self.annotations.append(dataset['annotations'][k])          
          
          self.dataset.append(data[i: i+len])
       
      test = 10
      
      self.ids = torch.Tensor(self.ids)[:test]
      self.dataset = torch.Tensor(self.dataset)[:test]
      self.annotations = torch.Tensor(self.annotations)[:test]

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, idx):
      input = self.dataset[idx].long()
      target = self.dataset[idx].long()

      return {'ids': self.ids[idx],  'inputs': input, 'targets': target, 'annotations': self.annotations[idx]}
