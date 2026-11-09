import numpy as np

def double_gyre_flow(amplitude, frequency, x, y, t):
    '''
    Solve the double gyre flow problem

    Inputs
        amplitude                   (`float`)
        frequency                   (`float`)
        horizontal discretization   (`np.array[float]`, shape: (ny,))
        vertical discretization     (`np.array[float]`, shape: (nx,))
        time vector                 (`np.array[float]`, shape: (ntimes,))

    Output
        horizontal velocity matrix  (`np.array[float]`, shape: (ntimes, nx * ny)
        vertical velocity matrix    (`np.array[float]`, shape: (ntimes, nx * ny)
    '''

    xgrid, ygrid = np.meshgrid(x, y)         # spatial grid

    u = np.zeros((len(t), len(x), len(y)))   # horizontal velocity
    v = np.zeros((len(t), len(x), len(y)))   # vertical velocity

    intensity = 0.1   # intensity parameter

    f = lambda x,t: amplitude * np.sin(frequency * t) * x**2 + x - 2 * amplitude * np.sin(frequency * t) * x

    # compute solution
    for i in range(len(t)):
        u[i] = (-np.pi * intensity * np.sin(np.pi * f(xgrid, t[i])) * np.cos(np.pi * ygrid)).T
        v[i] = (np.pi * intensity * np.cos(np.pi * f(xgrid, t[i])) * np.sin(np.pi * ygrid) * (2 * amplitude * np.sin(frequency * t[i]) * xgrid + 1.0 - 2 * amplitude * np.sin(frequency * t[i]))).T
    
    # compute analitical derivative
    du = np.zeros_like(u)
    dv = np.zeros_like(v)

    for i in range(len(t)):

        sinwt = np.sin(frequency * t[i])
        coswt = np.cos(frequency * t[i])

        f_xt = amplitude * sinwt * xgrid**2 + xgrid - 2 * amplitude * sinwt * xgrid
        dfdt = amplitude * frequency * coswt * (xgrid**2 - 2 * xgrid)

        g = 2 * amplitude * sinwt * xgrid + 1.0 - 2 * amplitude * sinwt
        dgdt = 2 * amplitude * frequency * coswt * (xgrid - 1)

        du[i] = (
            -np.pi**2 * intensity
            * np.cos(np.pi * f_xt)
            * np.cos(np.pi * ygrid)
            * dfdt
        ).T

        dv[i] = (
            np.pi * intensity * np.sin(np.pi * ygrid)
            * (
                -np.pi * np.sin(np.pi * f_xt) * dfdt * g
                + np.cos(np.pi * f_xt) * dgdt
            )
        ).T

    return u, v, du, dv


def create_data(amplitude, frequency, Lx, Ly, nx, ny, T, dt):

    # spatial discretization
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # temporal discretization
    t = np.arange(0, T + dt, dt)

    # get the ground-truth data from the solver
    u, v, du, dv = double_gyre_flow(amplitude, frequency, x, y, t)

    return u, v, du, dv, t, x, y



def get_dg_data(randomized=True, amplitude=0.25, frequency=5.0, Lx=2.0, Ly=1.0, nx=50, ny=25, T=10.0, dt=0.1):

    # 1. Generate data
    u, v, du, dv, t, x, y = create_data(amplitude, frequency, Lx, Ly, nx, ny, T, dt)

    # 2. Split training - validation - test
    n_samples = t.shape[0]

    if not randomized: # -> split: first 80% train, second 10% val, last 10% test
        training_samples = np.arange(int(.8*n_samples))
        val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
        test_samples = np.arange(int(.9*n_samples), n_samples)
    else: # -> permute the first 90%, set 80% train and 10% val, then last 10% test
        perm = np.random.permutation(int(.9*n_samples))
        training_samples = perm[:int(.8*n_samples)]
        val_samples = perm[int(.8*n_samples):]
        test_samples = np.arange(int(.9*n_samples), n_samples)

    # 3. Collect data in a dictionary
    training_data = {}
    validation_data = {}
    test_data = {}

    # utility function to stack x <- (u,v) and dx <- (du, dv)
    pack = lambda uu, vv: np.stack((uu, vv), axis=-1).astype(np.float32)

    training_data['x'] = pack(u[training_samples], v[training_samples])
    training_data['dx'] = pack(du[training_samples], dv[training_samples])
    training_data['t'] = t[training_samples].astype(np.float32)

    validation_data['x'] = pack(u[val_samples], v[val_samples])
    validation_data['dx'] = pack(du[val_samples], dv[val_samples])
    validation_data['t'] = t[val_samples].astype(np.float32)

    test_data['x'] = pack(u[test_samples], v[test_samples])
    test_data['dx'] = pack(du[test_samples], dv[test_samples])
    test_data['t'] = t[test_samples].astype(np.float32)

    # add in every dictionary also the spatial coordiantes
    for d in [training_data, validation_data, test_data]:
        d['x_grid'] = x
        d['y_grid'] = y

    return training_data, validation_data, test_data



### Torch loaders
import torch
from torch.utils.data import TensorDataset, DataLoader

# def get_dg_loaders(train_dict, val_dict, test_dict, batch_size=32, channels_first=True):
#     """
#     Trasforma i dizionari di Double Gyre in DataLoader pronti per il training.
    
#     Args:
#         train_dict, val_dict, test_dict: I dizionari in uscita da get_dg_data
#         batch_size (int): Dimensione del batch
#         channels_first (bool): Se True, trasforma (H, W, 2) in (2, H, W) per CNN
#     """
    
#     loaders = []
    
#     for d in [train_dict, val_dict, test_dict]:
#         # Conversione in Tensori
#         x = torch.from_numpy(d['x'])   # shape: [N, H, W, 2]
#         dx = torch.from_numpy(d['dx']) # shape: [N, H, W, 2]
#         # t = torch.from_numpy(d['t'])   # shape: [N] NOTE: Per ora non serve il tempo
        
#         # Se usi una CNN, PyTorch vuole i canali (u,v) prima delle dimensioni spaziali
#         if channels_first:
#             x = x.permute(0, 3, 1, 2)   # diventa: [N, 2, H, W]
#             dx = dx.permute(0, 3, 1, 2) # diventa: [N, 2, H, W]
        
#         # Creazione Dataset e Loader
#         # Poiché i dati sono già stati rimescolati in get_dg_data, 
#         # shuffle=False qui è sicuro e leggermente più veloce.
#         # ds = TensorDataset(x, dx, t)
#         ds = TensorDataset(x, dx)
#         loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
#         loaders.append(loader)
        
#     return loaders # Ritorna (train_loader, val_loader, test_loader)


def get_dg_loaders(train_dict, val_dict, test_dict, batch_size=32, channels_first=True, train_shuffle=False, include_time=False):
    """
    Transform the dictionaries of the Double Gyre problem in Torch loaders.
    
    Args:
        train_dict, val_dict, test_dict: dictionaries form get_dg_data
        batch_size (int): batch size
        channels_first (bool): if True, transform (H, W, 2) in (2, H, W) for CNN
        train_shuffle (bool): if True, shuffle the training data
        include_time (bool): if True, return (x, dx, t), otherwise (x, dx)
    """
    
    loaders = []

    for d in [train_dict]:
        # tensor conversions
        x = torch.from_numpy(d['x'])   # shape: [N, H, W, 2]
        dx = torch.from_numpy(d['dx']) # shape: [N, H, W, 2]
        
        # usually, CNN in PyTorch wants (u,v) before the spatial coordinates
        if channels_first:
            x = x.permute(0, 3, 1, 2)   # becomes: [N, 2, H, W]
            dx = dx.permute(0, 3, 1, 2) # becomes: [N, 2, H, W]
        
        # handle variable time
        if include_time:
            t = torch.from_numpy(d['t']) # shape: [N]
            ds = TensorDataset(x, dx, t) # dataset with time
        else:
            ds = TensorDataset(x, dx) # dataset without time
            
        # Final Loader
        if train_shuffle:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        loaders.append(loader)
    
    for d in [val_dict, test_dict]:
        # tensor conversions
        x = torch.from_numpy(d['x'])   # shape: [N, H, W, 2]
        dx = torch.from_numpy(d['dx']) # shape: [N, H, W, 2]
        
        # usually, CNN in PyTorch wants (u,v) before the spatial coordinates
        if channels_first:
            x = x.permute(0, 3, 1, 2)   # becomes: [N, 2, H, W]
            dx = dx.permute(0, 3, 1, 2) # becomes: [N, 2, H, W]
        
        # handle variable time
        if include_time:
            t = torch.from_numpy(d['t']) # shape: [N]
            ds = TensorDataset(x, dx, t) # dataset with time
        else:
            ds = TensorDataset(x, dx) # dataset without time
            
        # Final Loader
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        loaders.append(loader)
        
    return loaders # return (train_loader, val_loader, test_loader)