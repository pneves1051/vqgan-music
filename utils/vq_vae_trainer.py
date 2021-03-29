import torch
import time
import numpy as np
import defaultdict
import IPython.display as ipd

class VQVAETrainer():
  def __init__(self, vqvae, discriminator, transformer, dataloader, vqvae_loss, gan_loss, device, v_lr, d_lr, b_size, noise_dims, disc_steps=1):
    self.vqvae = vqvae
    self.discriminator = discriminator    
    self.dataloader = dataloader
    self.vqvae_loss = vqvae_loss
    self.gan_loss = gan_loss
    self.device=device
    self.b_size = b_size
    self.noise_dims = noise_dims
    self.disc_steps = disc_steps
    
    # betas=(0.5, 0.999)
    self.v_optimizer = torch.optim.Adam(self.vqvae.parameters(), lr = v_lr)
    self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = d_lr)
    #self.e_scheduler = torch.optim.lr_scheduler.StepLR(self.e_optimizer, 1.0, gamma=0.95) 
    #self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, 1.0, gamma=0.95) 
    #self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, 1.0, gamma=0.95) 
    self.real_label = 0.9
    self.fake_label = 0.

    # noise fixo para avaliação do modelo
    self.fixed_noise = torch.rand(1, self.noise_dims[0], self.noise_dims[1], device=device)
    
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
      annotations = data['annotations'].to(self.device)

      outputs, codes = self.vqvae(inputs, annotations)
      
      # loss, loss_list = self.loss_fn(outputs, targets, top_codes, bottom_codes)
      loss,_ = self.vqvae_loss(outputs, targets, codes)
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
    
      # Allow D to be updated
      
      ##########      
      # D update
      ##########      
      for p in self.discriminator.parameters():
        p.requires_grad = True
      
      for _ in range(self.disc_steps):

        self.discriminator.zero_grad()
        #Forward of the real batch through D
        d_real = self.discriminator(real)
        #torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 0.5)
        D_x = d_real.mean().item()

        # VQ_VAE reconstructions
        reconstructed, codes = self.vqvae(real)
        #reconstructed_gan, _, _ = self.generator(top_encoding.detach(), bottom_encoding.detach())

        # Fake batch goes through d
        d_fake = self.discriminator(reconstructed.detach())
        D_G_z1 = d_fake.mean().item()

        #Cálculo do erro no batch de amostras reais
        d_loss = self.gan_loss(d_real, d_fake, mode='d')

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
      #self.discriminator.zero_grad()
     
      d_fake = self.discriminator(reconstructed)
      D_G_z2 = d_fake.mean().item()       
        
      vae_loss, vqvae_loss_list = self.vqvae_loss(real, reconstructed, codes)
      #Calculamos o erro de G com base nesse output
      g_loss = self.gan_loss(d_fake, mode='g')
      #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA      
      (vae_loss + g_loss).backward()
      # Atualizamos G e E
      self.v_optimizer.step()
           
      #Estatísticas de treinamento        
      log_interval = 20
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        print('{:3d} batches | time: {:5.2f}s | Loss_D: {:5.4f} | Loss_G: {} | VQ_VAE_loss: {} | '
              ' | D(x): {:5.4f} | D(G(z)): {:5.4f} / {:5.4f} ' 
              .format(index, elapsed, d_loss.item(), g_loss.item(),
                      vqvae_loss_list, D_x, D_G_z1, D_G_z2))              
        start_time = time.time()

      # Salva Losses para plotar depois
      d_losses.append(d_loss.item())
      g_losses.append(g_loss.item())
      vqvae_losses.append(vqvae_loss_list)
      
    return np.mean(d_losses, 0), np.mean(g_losses, 0), np.mean(vqvae_losses, 0)

  def train(self, EPOCHS, checkpoint_dir, fixed_noise = None, train_gan=False):
    history = defaultdict(list)
    best_d_loss = 0.
    best_g_loss = 0.

    #Vetor para armazenar resultados do gerador
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
    with torch.no_grad():
      fake = self.vqvae(noise)[0].squeeze(0).detach().cpu()
      ipd.Audio(fake, rate=44100)
      return fake
    
  def save_model(self, checkpoint_dir):
      torch.save(self.vqvae.state_dict(), checkpoint_dir + 'test_vqvae_state.bin')    
      torch.save(self.discriminator.state_dict(), checkpoint_dir + 'test_discriminator_state.bin')
