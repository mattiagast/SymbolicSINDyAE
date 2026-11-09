import os
import numpy as np
import pickle
import torch

from autoencoder import SINDyConvAE


def train_sindy_convae(model, train_loader, val_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    print("\n[INFO] SINDy Autoencoder Configuration:")
    print(f"  > Decoder weight:  {model.loss_weights['decoder']:.2e}")
    print(f"  > SINDy x weight:  {model.loss_weights['sindy_x']:.2e}")
    print(f"  > SINDy z weight:  {model.loss_weights['sindy_z']:.2e}")
    print(f"  > SINDy Reg (L1):  {model.loss_weights['sindy_reg']:.2e}")
    
    # 1. Calcolo Norme Validation per Ratios (Baseline)
    # Simile a np.mean(val_data**2) del tuo codice TF
    print("[INFO] Computation norms on validation set...")
    val_x_accum, val_dx_accum, n_val_samples = 0, 0, 0
    for x, dx, t in val_loader:
        val_x_accum += torch.sum(x**2).item()
        val_dx_accum += torch.sum(dx**2).item()
        n_val_samples += x.size(0) * x.size(1) * x.size(2) * x.size(3) # numel totali
    
    # MSE di un predittore "zero" (media dei quadrati)
    val_x_norm = val_x_accum / n_val_samples
    val_dx_norm = val_dx_accum / n_val_samples
    print(f"[INFO] Validation X norm (MSE baseline): {val_x_norm:.4e}")
    print(f"[INFO] Validation dX norm (MSE baseline): {val_dx_norm:.4e}")

    # ==========================
    # PHASE 1: MAIN TRAINING
    # ==========================
    print("\n--- PHASE 1: TRAINING ---")
    
    for epoch in range(params['max_epochs']):
        model.train()
        train_metrics = {'decoder': 0, 'sindy_x': 0, 'sindy_z': 0, 'sindy_reg': 0}
        total_train_loss = 0
        
        # Loop Training
        for x, dx, t in train_loader: # Spacchettamento di t
            x, dx, t = x.to(device), dx.to(device), t.to(device)
            optimizer.zero_grad()

            x_rec, dx_rec, z, dz_true, dz_pred = model(x, dx, t)
            loss, metrics = model.get_loss_components(x, dx, x_rec, dx_rec, dz_true, dz_pred)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            for k, v in metrics.items(): train_metrics[k] += v

        # Loop Validation
        model.eval()
        val_metrics = {'decoder': 0, 'sindy_x': 0, 'sindy_z': 0, 'sindy_reg': 0}
        total_val_loss = 0
        with torch.no_grad():
            for x, dx, t in val_loader: # <--- Aggiunto t
                x, dx, t = x.to(device), dx.to(device), t.to(device)
                # Passiamo t al forward
                x_rec, dx_rec, z, dz_true, dz_pred = model(x, dx, t)
                loss, metrics = model.get_loss_components(x, dx, x_rec, dx_rec, dz_true, dz_pred)

                total_val_loss += loss.item()
                for k, v in metrics.items(): val_metrics[k] += v

        # Medie
        n_tr = len(train_loader)
        n_val = len(val_loader)
        
        avg_tr_loss = total_train_loss / n_tr
        for k in train_metrics: train_metrics[k] /= n_tr
        
        avg_val_loss = total_val_loss / n_val
        for k in val_metrics: val_metrics[k] /= n_val

        # LOGGING (ogni print_frequency)
        if epoch % params['print_frequency'] == 0:
            print(f"[TRAIN] Epoch {epoch:4d}")
            print(f"  Train Loss: {avg_tr_loss:.4e} ( Dec: {train_metrics['decoder']:.2e} | dX: {train_metrics['sindy_x']:.2e} | dZ: {train_metrics['sindy_z']:.2e} | Reg: {train_metrics['sindy_reg']:.2e} )")
            print(f"  Valid Loss: {avg_val_loss:.4e} ( Dec: {val_metrics['decoder']:.2e} | dX: {val_metrics['sindy_x']:.2e} | dZ: {val_metrics['sindy_z']:.2e} )")
            print(f"  >> RATIO: Decoder: {val_metrics['decoder']/val_x_norm:.4f} | SINDy: {val_metrics['sindy_x']/val_dx_norm:.4f}")

        # THRESHOLDING SEQUENZIALE
        if params.get('sequential_thresholding', False) and epoch > 0 and epoch % params.get('threshold_frequency', 10) == 0:
            with torch.no_grad():
                # Calcolo nuova maschera
                curr_coeffs = model.sindy_coeffs
                new_mask = (torch.abs(curr_coeffs) > params['coefficient_threshold']).float()
                
                # Aggiorno il buffer della maschera
                model.sindy_mask.data = new_mask
                
                # IMPORTANTE: Azzero fisicamente i coefficienti nel parametro per pulizia
                model.sindy_coeffs.data = curr_coeffs * new_mask
                
                active_count = torch.sum(new_mask).item()
                print(f"    >> THRESHOLDING: Active coefficients: {int(active_count)}/{new_mask.numel()}")

    # ==========================
    # PHASE 2: REFINEMENT
    # ==========================
    print("\n--- PHASE 2: REFINEMENT ---")
    # Spegniamo la regolarizzazione per il raffinamento
    model.loss_weights['sindy_reg'] = 0.0
    
    for epoch in range(params.get('refinement_epochs', 0)):
        model.train()
        train_metrics = {'decoder': 0, 'sindy_x': 0, 'sindy_z': 0, 'sindy_reg': 0}
        total_train_loss = 0
        
        for x, dx, t in train_loader:
            x, dx, t = x.to(device), dx.to(device), t.to(device)
            optimizer.zero_grad()
            
            x_rec, dx_rec, z, dz_true, dz_pred = model(x, dx, t)
            loss, metrics = model.get_loss_components(x, dx, x_rec, dx_rec, dz_true, dz_pred)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            for k, v in metrics.items(): train_metrics[k] += v

        # Loop Validation per Refinement
        model.eval()
        val_metrics = {'decoder': 0, 'sindy_x': 0, 'sindy_z': 0, 'sindy_reg': 0}
        total_val_loss = 0
        with torch.no_grad():
            for x, dx, t in val_loader:
                x, dx, t = x.to(device), dx.to(device), t.to(device)
                x_rec, dx_rec, z, dz_true, dz_pred = model(x, dx, t)
                loss, metrics = model.get_loss_components(x, dx, x_rec, dx_rec, dz_true, dz_pred)
                total_val_loss += loss.item()
                for k, v in metrics.items(): val_metrics[k] += v

        # Calcolo Medie
        n_tr = len(train_loader)
        n_val = len(val_loader)
        
        avg_tr_loss = total_train_loss / n_tr
        avg_val_loss = total_val_loss / n_val
        for k in train_metrics: train_metrics[k] /= n_tr
        for k in val_metrics: val_metrics[k] /= n_val

        # LOGGING (uniformato alla Fase 1)
        if epoch % params['print_frequency'] == 0:
            print(f"[REFINE] Epoch {epoch:4d}")
            print(f"  Train Loss: {avg_tr_loss:.4e} ( Dec: {train_metrics['decoder']:.2e} | dX: {train_metrics['sindy_x']:.2e} | dZ: {train_metrics['sindy_z']:.2e} )")
            print(f"  Valid Loss: {avg_val_loss:.4e} ( Dec: {val_metrics['decoder']:.2e} | dX: {val_metrics['sindy_x']:.2e} | dZ: {val_metrics['sindy_z']:.2e} )")
            print(f"  >> RATIO: Decoder: {val_metrics['decoder']/val_x_norm:.4f} | SINDy: {val_metrics['sindy_x']/val_dx_norm:.4f}")

    print("\n--- TRAINING COMPLETED ---")
    return model




def save_model_and_params(model, params, folder_path, model_name="sindy_ae"):
    """
    Salva i pesi del modello PyTorch, la maschera SINDy e i parametri.
    """
    os.makedirs(folder_path, exist_ok=True)

    # 1. Salva lo state_dict (Pesi + Buffer come la sindy_mask)
    # Usiamo l'estensione .pt o .pth standard per PyTorch
    weights_path = os.path.join(folder_path, f"{model_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"[INFO] PyTorch weights and buffers saved in: {weights_path}")
    
    # 2. Salva i parametri (dizionario) in pickle
    params_path = os.path.join(folder_path, f"{model_name}_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    print(f"[INFO] Parameters saved in: {params_path}")
    
    # 3. Opzionale: Salva i coefficienti SINDy finali in formato leggibile (CSV o NumPy)
    # Molto utile per analisi rapida senza ricaricare tutto il modello
    coeffs_path = os.path.join(folder_path, f"{model_name}_coefficients.npy")
    with torch.no_grad():
        final_xi = (model.sindy_coeffs * model.sindy_mask).cpu().numpy()
        np.save(coeffs_path, final_xi)
    print(f"[INFO] Final SINDy coefficients saved in: {coeffs_path}")




def load_model_and_params(folder_path, model_name="sindy_ae"):
    # 1. Carica i parametri
    params_path = os.path.join(folder_path, f"{model_name}_params.pkl")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    # 2. Inizializza l'architettura (deve essere la stessa classe)
    model = SINDyConvAE(params)
    
    # 3. Carica i pesi
    weights_path = os.path.join(folder_path, f"{model_name}.pth")
    model.load_state_dict(torch.load(weights_path))
    model.eval() # Imposta in modalità valutazione
    
    return model, params

