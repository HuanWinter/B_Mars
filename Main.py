from scipy.io import readsav
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import h5py
from ipdb import set_trace as st
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from funs import * 

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
train_filename = 'Mars.save'
valid1_filename = 'mvn_orbits_20161002T0749_20161003T0220.save'
valid2_filename = 'mvn_orbits_20161204T2209_20161205T1158.save'
Data_file = 'Data/ML_ready_data_2016_all.h5'

test_clu = ['test1', 'test2']
# Valid_file = 'Data/ML_ready_data_2016case.h5'
save_h5 = 'Res/results_2016case.h5'
figname = 'Figs/sza_vs_alt_2016case.png'

Overwrite = False # Reprocess the ML-ready data 
train_flag = False # retrain the MLP model
# Read or save ML-ready dataset
if not os.path.exists('Figs/'):
    os.makedirs('Figs/')

if not os.path.exists('Res/'):
    os.makedirs('Res/')
    
if ((os.path.exists(Data_file)==0) | (Overwrite==True)):
    stats = ML_dataset(train_filename, Data_file, mode='train')
    ML_dataset(valid1_filename, Data_file, stats=stats, mode='test1')
    ML_dataset(valid2_filename, Data_file, stats=stats, mode='test2')
        
with h5py.File(Data_file, 'r') as f:
    # print(f.keys())
    # st()
    X_train = np.array(f['X_train'])
    Y_train = np.array(f['Y_train'])
    
    X_test = []
    Y_test = []
    leng = np.zeros(len(test_clu)) # record the number of index from different events
    
    for idx, test_idx in enumerate(test_clu):    
        X_test.append(np.array(f['X_'+test_idx]))
        Y_test.append(np.array(f['Y_'+test_idx]))
        leng[idx] = np.array(f['Y_'+test_idx]).shape[0]
    
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    
    
st()
# Normalization 
# X_norm, X_mean, X_std = zscore_normalize(X_clean)
# Y_norm, Y_mean, Y_std = zscore_normalize(Y_clean)

# Convert to torch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train, dtype=torch.float32)

X_tensor_val = torch.tensor(X_test, dtype=torch.float32)
Y_tensor_val = torch.tensor(Y_test, dtype=torch.float32)
'''

X_train, y_train = X_tensor, Y_tensor
'''
# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X_tensor, 
                                                  Y_tensor, 
                                                  test_size=0.2, 
                                                  random_state=42)

_, X_test, _, y_test = train_test_split(X_tensor_val, 
                                        Y_tensor_val, 
                                        test_size=0.99, 
                                        random_state=42,
                                        shuffle=False
                                        )


train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
# test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, 
                          batch_size=256, 
                          num_workers=12,
                          shuffle=True)
val_loader = DataLoader(val_dataset, 
                        batch_size=1024,
                        num_workers=12)
test_loader = DataLoader(test_dataset, 
                        batch_size=1024,
                        num_workers=96)

################################## pdf weighted results

# Compute magnitude
mag = np.sqrt(np.mean(Y_train**2, axis=1))  # shape: (N, 3)

# Estimate PDF via histogram
hist, bin_edges = np.histogram(mag, bins=100, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Convert PDF to weights (inverse PDF for rare emphasis)
pdf_interp = interp1d(bin_centers, hist, kind='linear', 
                      bounds_error=False, fill_value='extrapolate')

def main():
    
    ############### Gradient Boosting ######################
    # Y_pred_GB = train_gradient_boosting(X_clean, Y_clean)
    
    ############### MLP Boosting ######################
    V_checkpoint_init = 'Checkpoints/test_init.ckpt'
    V_checkpoint_name = 'Checkpoints/test.ckpt'
    checkpoint_dir = 'Checkpoints/'
    if ((os.path.exists(V_checkpoint_name)) & (train_flag)):
        # st()
        os.remove(V_checkpoint_name)
        pass
    
    # st()
    
    model = MLPRegressor(input_dim=X_tensor.shape[1], 
                         hidden_dims=[128, 64], 
                         lr=1e-4,
                         pdf_interp=pdf_interp)

    bar = TQDMProgressBar()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='test',
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor='valid_loss',
        patience=20
    )

    trainer = Trainer(
        strategy="ddp",
        accelerator="gpu",
        devices=8,
        # devices="auto",
        max_epochs=1000,
        logger=True,
        deterministic=True,
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        callbacks=[
            bar,
            checkpoint_callback,
            early_stopping_callback
        ],
        precision=32
    )

    if train_flag:
        checkpoint = torch.load(V_checkpoint_init, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    
    checkpoint = torch.load(V_checkpoint_name, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to('cuda')
    model.eval()
    
    # pred_Y_test = torch.zeros(X_val.shape[0])  # Adjust output_dim accordingly
    # Y_clu = torch.zeros(X_val.shape[0])
    # X_clu = torch.zeros([X_tensor.shape[0], X_tensor.shape[1]])
    
    pred_Y_test = model(X_tensor_val.to('cuda')).cpu()
        
    with h5py.File(save_h5, 'w') as f:
        print(X_tensor_val.shape)
        print(X_tensor_val.shape)
        print(pred_Y_test.shape)
        f.create_dataset('X', data=X_tensor_val)
        f.create_dataset('Y', data=Y_tensor_val)
        f.create_dataset('Y_pred_MLP', data=pred_Y_test.detach().numpy())
        f.create_dataset('len', data=leng)
        
        # f.create_dataset('Y_pred_GB', data=Y_pred_GB.numpy())
    

if __name__ == "__main__":
    main()
