import random
import os
import pandas as pd
import torchaudio
import torch
import glob
import math

class AudioDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, annotations_path, sr, window_size, hop_length, batch_size):
    """
    Args:
      dataset_dir (string): dataset directory
      sr: sample rate of the audio
      window_size: number of samples of each item
      hop_length: step size
      batch_size: batch_size
    """
    self.dataset_dir = dataset_dir
    self.file_list = glob.glob(dataset_dir)
    annotations_df = pd.read_csv(annotations_path)
    annotations_df.rename(columns = {
          'musicId': 'id',
          'Arousal(mean)': 'arousal',
          'Valence(mean)': 'valence'
          }, inplace=True)
    self.annotations = annotations_df.to_numpy()

    self.sr = sr
    self.window_size = window_size
    self.hop_length = hop_length
    self.batch_size = batch_size

    self.start = 0
    self.end = len(self.file_list)
    
  @property
  def shuffled_file_list(self):
    shuffled_list = self.file_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

  def read_dataset(self, file_list, batch_size):
    shuffled_list = file_list.copy()
    random.shuffle(shuffled_list)    
    for file in shuffled_list:
      ids=[]
      audio=[]
      annotations=[]
      
      music_id = os.path.basename(str(file)).replace('.mp3', '')
      if music_id != '' and np.any(float(music_id) in self.annotations[:, 0]):
        
        # carrega arquivo de áudio
        signal, orig_sr = torchaudio.load(file)
        signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
        # normalização entre [-1, 1]
        signal = signal - signal.mean()
        signal = signal/signal.abs().max()
        signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        assert not torch.any(signal.abs() > 1.)
        signal = signal[:1, :]

        # anotações de alerta e valência
        annotation = torch.Tensor([a[1:] for a in self.annotations if int(a[0]) == int(music_id)])
              
        for j in range(0, signal.shape[1] -self.window_size, self.hop_length):
          
          current_signal = signal[:, j: j+self.window_size]
          ids.append(torch.Tensor([int(music_id)]))
          audio.append(current_signal)
          #####ATTENTION: NOT USING WHOLE SLICE
          annotations.append(annotation)

          if len(audio) >= batch_size:
            data_batch = {'ids': [], 'inputs': [], 'targets': [], 'annotations': []}
            data_batch['ids'] = torch.stack(ids)
            data_batch['inputs'] =  torch.stack(audio)
            data_batch['targets'] = torch.stack(audio)
            data_batch['annotations'] = torch.stack(annotations)
            
            ids=[]
            audio = []
            annotations = []
            
            yield data_batch        

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      iter_start = self.start
      iter_end = self.end
    else:
      # divide carga de trabalho
      per_worker = int(math.ceil((self.end-self.start)/ float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = self.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return self.read_dataset(self.file_list[iter_start: iter_end],self.batch_size)
    #return self.get_streams()
    #return self.get_batch(self.shuffled_file_list)
    #return self.read_dataset(self.shuffled_file_list, self.batch_size)
    
