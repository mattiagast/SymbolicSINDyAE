import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import time




def set_global_seed(seed: int = 42):
    # Python base
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (Sostituisce la parte TensorFlow)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Se usi più GPU
    
    # Operazioni Deterministiche (Fondamentale per la riproducibilità)
    # Equivalente a TF_DETERMINISTIC_OPS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Forza PyTorch a usare algoritmi deterministici dove possibile
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[INFO] Global seed set to {seed} (PyTorch, NumPy, Random)")




set_global_seed(42)



class SINDyConvAE(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.library_dim = params['library_dim']
        self.poly_order = params.get('poly_order', 2)
        self.include_sine = params.get('include_sine', False)
        self.act_type = params.get('activation', 'tanh')
        
        # Pesi della Loss (mutabili per la Phase 2)
        self.loss_weights = params.get('loss_weights', {
            'decoder': 1.0, 'sindy_x': 1e-4, 'sindy_z': 0.1, 'sindy_reg': 1e-5
        })

        # --- ENCODER ---
        enc_layers = []
        in_ch = params['input_shape'][-1]
        for nf in params['filter_widths']:
            enc_layers.append(nn.Conv2d(in_ch, nf, kernel_size=3, stride=2, padding=1))
            in_ch = nf
        self.enc_convs = nn.ModuleList(enc_layers)
        
        # Dimensione flat
        dummy_input = torch.zeros(1, params['input_shape'][-1], params['input_shape'][0], params['input_shape'][1])
        with torch.no_grad():
            self.flat_dim = self._get_enc_output_shape(dummy_input).numel()
        
        self.dense_z = nn.Linear(self.flat_dim, self.latent_dim)

        # --- SINDy ---
        # Inizializzazione
        # self.sindy_coeffs = nn.Parameter(torch.randn(self.library_dim, self.latent_dim))
        self.sindy_coeffs = nn.Parameter(torch.randn(self.library_dim, self.latent_dim))# * 0.01)
        # Buffer per la maschera (non addestrabile, salvato nel state_dict)
        self.register_buffer('sindy_mask', torch.ones(self.library_dim, self.latent_dim))

        # --- DECODER ---
        self.dec_dense = nn.Linear(self.latent_dim, self.flat_dim)
        
        dec_layers = []
        rev_filters = params['filter_widths'][::-1]
        for i in range(len(rev_filters)-1):
            dec_layers.append(nn.ConvTranspose2d(rev_filters[i], rev_filters[i+1], kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dec_convs = nn.ModuleList(dec_layers)
        self.dec_out = nn.ConvTranspose2d(rev_filters[-1], params['input_shape'][-1], kernel_size=3, stride=2, padding=1, output_padding=1)

    # --- HELPER ---
    def _activation(self, x):
        if self.act_type == 'tanh': return torch.tanh(x)
        if self.act_type == 'elu': return F.elu(x)
        if self.act_type == 'relu': return F.relu(x)
        return x 

    def _activation_deriv(self, x, y):
        if self.act_type == 'tanh': return 1.0 - y**2
        if self.act_type == 'elu': return torch.where(x > 0, torch.ones_like(x), y + 1.0)
        if self.act_type == 'relu': return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        return torch.ones_like(x)

    def _get_enc_output_shape(self, x):
        for l in self.enc_convs: x = self._activation(l(x))
        return x

    def construct_sindy_library(self, z, t):
        # library = [tf.ones((tf.shape(z)[0], 1))] -> Traduzione PyTorch:
        batch_size = z.shape[0]
        device = z.device
        library = [torch.ones(batch_size, 1).to(device)]
        
        # Termini lineari
        for i in range(self.latent_dim):
            library.append(z[:, i:i+1])
            
        # Termini quadratici
        if self.poly_order > 1:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append(z[:, i:i+1] * z[:, j:j+1])
                    
        # Termini cubici
        if self.poly_order > 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    for k in range(j, self.latent_dim):
                        library.append(z[:, i:i+1] * z[:, j:j+1] * z[:, k:k+1])
                        
        # Termini trigonometrici
        if self.include_sine:
            # Invece di sin(z), aggiungiamo sin(5t)
            # Assicurati che t abbia shape (batch_size, 1)
            if t.dim() == 1: 
                t = t.view(-1, 1)
                # library.append(torch.cos(5.0 * t))
                library.append(torch.cos(4.998442071146355 * t))
                
        # return tf.concat(library, axis=1) -> Traduzione PyTorch:
        return torch.cat(library, dim=1)

    def propagate(self, x, dx, layers):
        z, dz = x, dx
        for layer in layers:
            z_pre = layer(z)
            dz_pre = layer(dz) - layer.bias.view(1, -1, 1, 1) # Bias correction for deriv
            z = self._activation(z_pre)
            dz = self._activation_deriv(z_pre, z) * dz_pre
        return z, dz

    def forward(self, x, dx, t):
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        dx = dx.permute(0, 3, 1, 2)
        
        # Encoder
        z_conv, dz_conv = self.propagate(x, dx, self.enc_convs)
        z_flat, dz_flat = torch.flatten(z_conv, 1), torch.flatten(dz_conv, 1)
        z = self.dense_z(z_flat)
        dz_true = self.dense_z(dz_flat) - self.dense_z.bias
        
        # SINDy - Ora passiamo anche t
        Theta = self.construct_sindy_library(z, t)
        Xi_eff = self.sindy_coeffs * self.sindy_mask
        dz_pred = torch.matmul(Theta, Xi_eff)
        
        # Decoder
        dec_flat_pre = self.dec_dense(z)
        dec_flat = self._activation(dec_flat_pre)
        
        dz_dec_flat_pre = self.dec_dense(dz_true) - self.dec_dense.bias
        dz_dec_flat = self._activation_deriv(dec_flat_pre, dec_flat) * dz_dec_flat_pre
        
        dec_conv_in = dec_flat.view(-1, *z_conv.shape[1:])
        dz_dec_conv_in = dz_dec_flat.view(-1, *dz_conv.shape[1:])
        
        x_rec_conv, dx_rec_conv = self.propagate(dec_conv_in, dz_dec_conv_in, self.dec_convs)
        
        x_rec = self.dec_out(x_rec_conv)
        dx_rec = self.dec_out(dx_rec_conv) - self.dec_out.bias.view(1, -1, 1, 1)
        
        return x_rec.permute(0, 2, 3, 1), dx_rec.permute(0, 2, 3, 1), z, dz_true, dz_pred

    def get_loss_components(self, x, dx, x_rec, dx_rec, dz_true, dz_pred):
        mse = F.mse_loss
        l_recon = mse(x_rec, x)
        l_dx = mse(dx_rec, dx)
        l_dz = mse(dz_pred, dz_true)
        # Regolarizzazione L1 sui coefficienti mascherati
        l_reg = torch.mean(torch.abs(self.sindy_coeffs * self.sindy_mask))
        
        w = self.loss_weights
        total = (w['decoder'] * l_recon + 
                 w['sindy_x'] * l_dx + 
                 w['sindy_z'] * l_dz + 
                 w['sindy_reg'] * l_reg)
                 
        return total, {'decoder': l_recon.item(), 'sindy_x': l_dx.item(), 
                       'sindy_z': l_dz.item(), 'sindy_reg': l_reg.item()}
    
    
    
    def encode(self, x):
        # Gestione shape (se arriva come NHWC)
        if x.shape[-1] == self.params['input_shape'][-1]:
            x = x.permute(0, 3, 1, 2)
        
        # Passaggio nei conv
        z_conv = x
        for l in self.enc_convs:
            z_conv = self._activation(l(z_conv))
        
        # Flatten e linear
        z_flat = torch.flatten(z_conv, 1)
        z = self.dense_z(z_flat)
        return z
    
    def get_z_conv_shape(self, device):
        """Metodo helper per calcolare la shape spaziale prima del flatten"""
        dummy_input = torch.zeros(
            1, 
            self.params['input_shape'][-1], 
            self.params['input_shape'][0], 
            self.params['input_shape'][1]
        ).to(device)
        with torch.no_grad():
            z_conv = dummy_input
            for l in self.enc_convs:
                z_conv = self._activation(l(z_conv))
            return z_conv.shape

    def decode(self, z):
        """Trasforma le coordinate latenti z in immagini ricostruite"""
        # 1. Dense layer iniziale del decoder
        dec_flat_pre = self.dec_dense(z)
        dec_flat = self._activation(dec_flat_pre)
        
        # 2. Reshape per tornare a un tensore spaziale (Convolutional)
        z_conv_shape = self.get_z_conv_shape(z.device)
        dec_conv_in = dec_flat.view(-1, *z_conv_shape[1:])
        
        # 3. Passaggio attraverso i layer ConvTranspose (Up-sampling)
        x_rec_conv = dec_conv_in
        for l in self.dec_convs:
            x_rec_conv = self._activation(l(x_rec_conv))
        
        # 4. Layer di uscita finale
        x_rec = self.dec_out(x_rec_conv)
        
        # 5. Permute per tornare al formato NHWC (necessario per visualizzare con plt.imshow)
        return x_rec.permute(0, 2, 3, 1)
    



