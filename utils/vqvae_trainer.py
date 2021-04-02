import torch
import time
import numpy as np
from collections import defaultdict
import IPython.display as ipd

class VQVAETrainer():
  def __init__(self, vqvae, discriminator, dataloader, vqvae_loss, gan_loss, hps, device):
    self.vqvae = vqvae
    self.discriminator = discriminator    
    self.dataloader = dataloader
    self.vqvae_loss = vqvae_loss
    self.gan_loss = gan_loss
    self.device=device
    
    self.spec_hp = hps['model']['vqgan']['vqvae']['loss']['spectral_hp']
    self.disc_steps = hps['model']['vqgan']['disc']['disc_steps']

    # betas=(0.5, 0.999)
    self.v_optimizer = torch.optim.Adam(self.vqvae.parameters(), 
                                        lr = float(hps['model']['vqgan']['vqvae']['lr']))
    self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                        lr = float(hps['model']['vqgan']['disc']['lr']))
    #self.e_scheduler = torch.optim.lr_scheduler.StepLR(self.e_optimizer, 1.0, gamma=0.95) 
    #self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, 1.0, gamma=0.95) 
    #self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, 1.0, gamma=0.95) 

    self.hps=hps
    
  def get_loss_hp(self, rec_loss, gan_loss, last_layer):
    rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

    disc_weight = torch.linalg.norm(rec_grads, 'fro') / (torch.linalg.norm(gan_grads, 'fro') + 1e-4)
    disc_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    
    return disc_weight  
  
  def train_epoch(self):
    self.vqvae.train()
    self.discriminator.train()
       
    losses = []
    losses_list = []
    start_time = time.time()
    index=0
    for data in self.dataloader:
      inputs = data['inputs'].to(self.device)
      targets = data['targets'].to(self.device)
      conditions = data['condiditions'].to(self.device)

      outputs, codes = self.vqvae(inputs, annotations)
      
      # loss, loss_list = self.loss_fn(outputs, targets, top_codes, bottom_codes)
      loss_l = self.vqvae_loss(outputs, targets, codes)
      loss = sum(loss_l)
      self.v_optimizer.zero_grad()
      loss.backward()
      #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)     
      self.v_optimizer.step()
      
      losses.append(sum(loss).item())
      losses_list.append(torch.Tensor(loss).tolist())

      index += 1        
      log_interval = 1
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        current_loss = np.mean(losses)
        print('| {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
              'loss {:5.5f} | loss_list: {}'.format(
              index, 2, #self.g_scheduler.get_last_lr()[0],
              elapsed*1000/log_interval,
              current_loss, np.mean(losses_list, axis=0)))
        start_time = time.time()

    train_loss = np.mean(losses)
    train_loss_list = np.mean(losses_list, axis=0)
    return train_loss, train_loss_list
    
  def train_epoch_gan(self):
    self.vqvae.train()
    self.discriminator.train()

    start_time = time.time()

    d_losses = []
    g_losses = []
    vqvae_losses = []
    g_codes_losses = []

    index=0
    for data in self.dataloader:
      index+=1
      
      #Treino com batch real
      real = data['inputs'].to(self.device)
      conditions = data['conditions']
      if conditions is not None:
        conditions = conditions.to(self.device)
          
      ##########      
      # D update
      ##########  
      # Allow D to be updated
          
      for p in self.discriminator.parameters():
        p.requires_grad = True
      
      for _ in range(self.disc_steps):

        self.discriminator.zero_grad()
        #Forward of the real batch through D
        d_real = self.discriminator(real, conditions)
        #torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 0.5)

        # VQ_VAE reconstructions
        fake, codes = self.vqvae(real)
        # Fake batch goes through d
        d_fake = self.discriminator(fake.detach())
        
        d_loss = 0
        D_x = 0
        D_G_z1 = 0
        for score_real, score_fake in zip(d_real, d_fake):
          D_x += d_real.mean().item()
          D_G_z1 += d_fake.mean().item()
          #Cálculo do erro no batch de amostras reais
          d_loss += self.gan_loss(score_real, score_fake, mode='d')

        #Calcula os gradientes para o batch
        d_loss.backward()
        #Atualza D
        self.d_optimizer.step()
     
      #########
      # G update
      #########
      # Stops D from being updated
      for p in self.discriminator.parameters():
        p.requires_grad = False      
      
      self.vqvae.zero_grad()
     
      fake, codes = self.vqvae(real)

      d_fake = self.discriminator(fake, conditions)
       
      rec_loss, lat_loss, spec_loss = self.vqvae_loss(real, fake, codes)
            
      g_loss = 0
      for score_fake in d_fake:
        D_G_z2 += score_fake.mean().item()
        # Calculate G loss
        g_loss += self.gan_loss(score_fake, mode='g')
      
      loss_hp = self.get_loss_hp(rec_loss, gan_loss, self.vq_vae.get_last_layer)
      #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA  
      total_loss = rec_loss + lat_loss + spec_loss + loss_hp*g_loss    
      total_loss.backward()
      # Atualizamos G e E
      self.v_optimizer.step()
           
      #Estatísticas de treinamento        
      log_interval = 20
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        print('{:3d} batches | time: {:5.2f}s | Loss_D: {:5.4f} | Loss_G: {} | VQ_VAE_loss: {} | '
              ' | D(x): {:5.4f} | D(G(z)): {:5.4f} / {:5.4f} ' 
              .format(index, elapsed, d_loss.item(), g_loss.item(),
                      [rec_loss.item(), lat_loss.item(), spec_loss.item()], D_x, D_G_z1, D_G_z2))              
        start_time = time.time()

      # Save losses to plot later
      d_losses.append(d_loss.item())
      g_losses.append(g_loss.item())
      vqvae_losses.append([rec_loss.item(), lat_loss.item(), spec_loss.item()])
      
    return np.mean(d_losses, 0), np.mean(g_losses, 0), np.mean(vqvae_losses, 0)

  def train(self, EPOCHS, checkpoint_dir, train_gan=False):
    history = defaultdict(list)
    best_d_loss = 0.
    best_g_loss = 0.

    # list to store generator outputs
    samples=[]

    for epoch in range(EPOCHS):
      epoch_start_time = time.time()
      print(f'Epoch {epoch + 1}/{EPOCHS}')

      print('-' * 10)
      if train_gan:
        d_loss, g_loss, vae_loss_list  = self.train_epoch_gan()
        
        print('| End of epoch {:3d}  | time: {:5.2f}s | loss_D: {:5.2f} |'
              'loss_G: {} | loss_VAE: {} '.format(
              epoch+1, (time.time()-epoch_start_time), d_loss, g_loss,
              vae_loss_list))
      else:
        train_loss, train_loss_list = self.train_epoch()
        history['train_loss'].append(train_loss)
        history['train_loss_list'].append(train_loss_list)
        #valid_loss, valid_loss_list = self.evaluate(self.valid_dataloader)

        print('| End of epoch {:3d}  | time: {:5.2f}s | train loss {:5.5f} | train_loss_list: {}'
              ' valid loss {} | valid_loss_list: {} '.format(epoch+1, (time.time()-epoch_start_time),
                                            train_loss, train_loss_list, 'valid_loss', 'valid_loss_list'))

      torch.save(self.vqvae.state_dict(), checkpoint_dir + 'test_vqvae_state.bin')
      torch.save(self.discriminator.state_dict(), checkpoint_dir + 'test_discriminator_state.bin')
      
      #self.e_scheduler.step()
      #self.d_scheduler.step()
      #self.g_scheduler.step()

      #fake = self.evaluate(self.top_fixed_noise, self.bottom_fixed_noise)
      #samples.append(fake)
                    
    return samples

  def evaluate(self, noise):
    data = next(iter(self.dataloader))
    real = data['inputs'].to(self.device)
    conditions = data['conditions']
    if conditions is not None:
      conditions = conditions.to(self.device)
    with torch.no_grad():
      fake = self.vqvae(real).detach().cpu()
      return fake
    
  def save_model(self, checkpoint_dir):
      torch.save(self.vqvae.state_dict(), checkpoint_dir + 'test_vqvae_state.bin')    
      torch.save(self.discriminator.state_dict(), checkpoint_dir + 'test_discriminator_state.bin')
