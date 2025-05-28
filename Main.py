from scipy.io import readsav
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import h5py
from ipdb import set_trace as st

from sklearn.model_selection import train_test_split
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from funs import * 

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4, 5, 6"
Data_file = 'Data/ML_ready_data.h5'
save_h5 = 'Res/results.h5'
figname = 'Figs/sza_vs_alt.png'

Overwrite = False
train_flag = False
# Read or save ML-ready dataset
if not os.path.exists('Figs/'):
    os.makedirs('Figs/')
    
if ((os.path.exists(Data_file)) & (Overwrite==False)):
    with h5py.File(Data_file, 'r') as f:
        X_clean = np.array(f['X'])
        Y_clean = np.array(f['Y'])
else:
    filename = 'Mars.save'
    data = readsav(filename)
    target = 'mse_b1'

    # st()
    # Concatenate into feature matrix X
    X = process_inputs_from_sav(data)
    Y = np.asarray(np.sqrt((np.sum(data[target]**2, axis=1))))
    print(f"Final shape of X: {X.shape}")  # Should be [N, num_total_features]
    print(f"Final shape of Y: {Y.shape}")  # Should be [N, num_total_features]

    # st()
    
    # Create a mask for rows without NaNs in X and Y
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)

    # Apply the mask to filter both X and Y
    X_clean = X[mask]
    Y_clean = Y[mask]

    print(f"Clean X shape: {X_clean.shape}")
    print(f"Clean Y shape: {Y_clean.shape}")

    with h5py.File(Data_file, 'w') as f:
        f.create_dataset('X', data=X_clean)
        f.create_dataset('Y', data=Y_clean)

# st()
# Normalization 
# X_norm, X_mean, X_std = zscore_normalize(X_clean)
# Y_norm, Y_mean, Y_std = zscore_normalize(Y_clean)

# Convert to torch tensors
X_tensor = torch.tensor(X_clean, dtype=torch.float32)
Y_tensor = torch.tensor(Y_clean, dtype=torch.float32)

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X_tensor, 
                                                  Y_tensor, 
                                                  test_size=0.2, 
                                                  random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_tensor, Y_tensor)

train_loader = DataLoader(train_dataset, 
                          batch_size=1024, 
                          num_workers=12,
                          shuffle=True)
val_loader = DataLoader(val_dataset, 
                        batch_size=1024,
                        num_workers=12)
test_loader = DataLoader(test_dataset, 
                        batch_size=1024,
                        num_workers=96)


def main():
    
    ############### Gradient Boosting ######################
    Y_pred_GB = train_gradient_boosting(X_clean, Y_clean)
    
    ############### MLP Boosting ######################
    V_checkpoint_init = 'Checkpoints/test_init.ckpt'
    V_checkpoint_name = 'Checkpoints/test.ckpt'
    checkpoint_dir = 'Checkpoints/'
    if ((os.path.exists(V_checkpoint_name)) & (train_flag)):
        # st()
        os.remove(V_checkpoint_name)
        pass
    
    model = MLPRegressor(input_dim=X_tensor.shape[1], 
                         hidden_dims=[128, 64], 
                         lr=1e-3)

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
        devices="auto",
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
    
    pred_Y_test = torch.zeros(X_tensor.shape[0])  # Adjust output_dim accordingly
    Y_clu = torch.zeros(X_tensor.shape[0])
    # X_clu = torch.zeros([X_tensor.shape[0], X_tensor.shape[1]])
    
    with torch.no_grad():
        offset = 0
        for batch in tqdm(test_loader):
            # st()
            X_batch = batch[0].to('cuda', non_blocking=True)  # Assuming your model is on the 'device' (e.g., 'cuda' or 'cpu')
            dst_preds = model(X_batch).cpu()
            B = batch[0].size(0)
            # st()
            pred_Y_test[offset:offset+B] = dst_preds
            Y_clu[offset:offset+B] = batch[1]
            
            offset += B
        
    with h5py.File(save_h5, 'w') as f:
        print(X_clean.shape)
        print(pred_Y_test.shape)
        f.create_dataset('X', data=X_clean)
        f.create_dataset('Y', data=Y_clu)
        f.create_dataset('Y_pred_MLP', data=pred_Y_test.numpy())
        f.create_dataset('Y_pred_GB', data=Y_pred_GB.numpy())
    

if __name__ == "__main__":
    main()
