import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

from datetime import datetime, timezone
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from scipy.io import readsav
from scipy.signal import butter, filtfilt

import h5py
from typing import Dict, Any

from ipdb import set_trace as st

########################## Basic function ####################

def remove_nan_rows_from_dict(data_dict: Dict[str, Any], count_key: str = 'npt') -> Dict[str, Any]:
    """
    Removes rows containing NaN values from a dictionary of synchronized NumPy arrays.

    This function iterates through all NumPy arrays in the input dictionary that
    have a length specified by `data_dict[count_key]`. It creates a master mask
    to identify every row index that contains a NaN in any of these arrays.
    It then returns a new dictionary where all corresponding arrays have been
    sliced to remove these rows, and the count value is updated.

    Args:
        data_dict (Dict[str, Any]): The input dictionary containing NumPy arrays and other data.
                                   It's expected that all arrays to be cleaned share the
                                   same first dimension length.
        count_key (str): The key in `data_dict` that holds the integer count of
                         rows/data points (e.g., 'npt').

    Returns:
        Dict[str, Any]: A new dictionary with NaN-containing rows removed from all
                        relevant arrays and the count key updated.

    Raises:
        ValueError: If the `count_key` is not found in the dictionary.
    """
    if count_key not in data_dict:
        raise ValueError(f"Error: The specified count_key '{count_key}' was not found in the dictionary.")

    n_points = data_dict[count_key]

    # Initialize a mask to mark every row that contains a NaN.
    # We start with all False (assuming all rows are good).
    combined_nan_mask = np.zeros(n_points, dtype=bool)

    # Iterate through all items to build the master NaN mask
    for key, value in data_dict.items():
        # Process only NumPy arrays that are part of the main dataset
        if isinstance(value, np.ndarray) and value.shape[0] == n_points:
            # For 2D+ arrays, check for NaNs across the columns (axis=1)
            if value.ndim > 1:
                current_mask = np.isnan(value).any(axis=1)
            # For 1D arrays
            else:
                current_mask = np.isnan(value)

            # Update the combined mask using a logical OR.
            # A row is marked bad (True) if it was already bad OR if it's bad in the current array.
            np.logical_or(combined_nan_mask, current_mask, out=combined_nan_mask)

    # Invert the mask to get all rows that are GOOD (do not have NaNs)
    good_indices_mask = ~combined_nan_mask
    new_n_points = good_indices_mask.sum()

    print(f"Original number of points: {n_points}")
    print(f"Number of rows with NaNs to be removed: {n_points - new_n_points}")
    print(f"New number of points after cleaning: {new_n_points}")

    # Build the new, cleaned dictionary
    cleaned_dict = {}
    for key, value in data_dict.items():
        if key == count_key:
            cleaned_dict[key] = new_n_points
        elif isinstance(value, np.ndarray) and value.shape[0] == n_points:
            # Slice the array to keep only the good rows
            cleaned_dict[key] = value[good_indices_mask]
        else:
            # Copy any other metadata that should not be sliced
            cleaned_dict[key] = value
            
    return cleaned_dict


def butterfilter(data,
                 cutoff=5,
                 order=4,
                 fs=100
                 ):
    
    # Design the low-pass filter
    # cutoff: Cutoff frequency in Hz
    # order: Filter order
    # fs: Sampling frequency in Hz

    # Get filter coefficients
    b, a = butter(order, cutoff, 
                    btype='low', 
                    analog=False, 
                    fs=fs
                    )

    # 3. Apply the filter to each dimension
    Y_filtered = filtfilt(b, a, data, axis=0)
    
    return Y_filtered

        
def to_native(array):
    """Convert big-endian numpy array to native byte order."""
    if isinstance(array, np.ndarray) and array.dtype.byteorder == '>':
        return array.byteswap().view(array.dtype.newbyteorder('='))
    return array

class ProgressBar_tqdm(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(            
            disable=True,            
        )
        return bar
    
def zscore_normalize(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std
    return X_norm, X_mean, X_std


def process_inputs_from_sav(data):
    # Create a DataFrame from scalar fields
    df = pd.DataFrame({
        'geo_longitude': data['geo_longitude'].flatten(),
        'geo_latitude': data['geo_latitude'].flatten(),
        'ls_mars': data['ls_mars'].flatten(),
        'subsolar_lng': data['subsolar_lng'].flatten(),
        'subsolar_lat': data['subsolar_lat'].flatten(),
        'mso_sza': data['mso_sza'].flatten(),
        'mse_azimuthal': data['mse_azimuthal'].flatten(),
        'altitude': data['altitude'].flatten(),
        'upstream_nsw': data['upstream_nsw'].flatten(),
        'f107_mars': data['f107_mars'].flatten(),
    })

    # Vector fields: shape should be (N, 3)
    df['upstream_imf'] = list(data['upstream_imf'])  # shape: (N, 3)
    df['upstream_usw'] = list(data['upstream_usw'])  # shape: (N, 3)

    # Reuse previous process_inputs function
    return process_inputs(df)

def ML_dataset(filename, Data_file, stats=None, mode='train', upstream_flag=False):
    
    data = readsav(filename)
    target = 'mse_b1'
    boundary = 'vignes_empirical_boundaries.save'
    
    # st()
    # --- 2. Call the function to clean the data ---
    print("--- Starting Cleaning Process ---")
    cleaned_data = remove_nan_rows_from_dict(data, count_key='npt')
    print("--- Cleaning Complete --- \n")

    # Create a mask for removing test set from the training set        
    if mode == 'train':
        
        # Concatenate into feature matrix X
        # X = process_inputs_from_sav(data)
        # st()
        
        X, stats, df = process_inputs(cleaned_data)
        Y = np.asarray(cleaned_data[target])
        
        # apply low-pass filter
        Y = butterfilter(Y)
        
        # st()
        # add the variation of mse_b1 as the weights during training
        Y_diff = np.diff(Y, axis=0, prepend=Y[:1, :])
        Y = np.hstack((Y, Y_diff))
        
        # Y = np.asarray(np.sqrt((np.sum(data[target]**2, axis=1))))
        print(f"Initial shape of X_train: {X.shape}")  # Should be [N, num_total_features]
        print(f"Initial shape of Y_train: {Y.shape}")  # Should be [N, num_total_features]
        
        # print(f"Final shape of Y_smooth: {Y_filtered.shape}")  # Should be [N, num_total_features]
        # st()
        # Define boundaries in Unix time
        start = datetime(2016, 10, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2016, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
        
        # st()
        # remove test set from the training
        time = cleaned_data['unixtime']
        mask_train = (time < start) | (time > end)  # Keep samples outside the specified range
        X = X[mask_train]
        Y = Y[mask_train]
        time = time[mask_train]
        
        print(f"Initial train shape of X_train: {X.shape}")  # Should be [N, num_total_features]
        print(f"Initial train shape of Y_train: {Y.shape}")  # Should be [N, num_total_features]
        
        if upstream_flag:
            ############################### remove upstream data  
            boundary_data = readsav(boundary)
            idx_sel = find_indices_below_mpb(df, boundary_data, mask_train)
            
            X = X[idx_sel]
            Y = Y[idx_sel]
            time = time[idx_sel]
        
        # Y = np.asarray(np.sqrt((np.sum(data[target]**2, axis=1))))
        print(f"Final nonupstream shape of X_train: {X.shape}")  # Should be [N, num_total_features]
        print(f"Final nonupstream shape of Y_train: {Y.shape}")  # Should be [N, num_total_features]
        
    else:
        X, _, df = process_inputs(cleaned_data, stats)
        Y = np.asarray(cleaned_data[target])
        Y = butterfilter(Y)
        
        # add the variation of mse_b1 as the weights during training
        Y_diff = np.diff(Y, axis=0, prepend=Y[:1, :])
        Y = np.hstack((Y, Y_diff))
        
        time = cleaned_data['unixtime']
        
        if upstream_flag:
            
            ############################### remove upstream data  
            boundary_data = readsav(boundary)
            idx_sel = find_indices_below_mpb(df, boundary_data, np.ones(X.shape[0]).astype(bool))
            
            X = X[idx_sel]
            Y = Y[idx_sel]
            time = time[idx_sel]
        
        # st()
        print(f"Final shape of X_test: {X.shape}")  # Should be [N, num_total_features]
        print(f"Final shape of Y_test: {Y.shape}")  # Should be [N, num_total_features]

    # Ensure native byte order
    X = to_native(X)
    Y = to_native(Y)
    
    # st()
    
    # Create a mask for rows without NaNs in X and Y
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)

    # Apply the mask to filter both X and Y
    X_clean = X[mask]
    Y_clean = Y[mask]
    Time_clean = time[mask]
    
    print(f"Clean X shape: {X_clean.shape}")
    print(f"Clean Y shape: {Y_clean.shape}")
    print(f"Clean Time shape: {Time_clean.shape}")
    # print(f"Clean Y shape: {Y_clean.shape}")
    
    with h5py.File(Data_file, 'a') as f:
        
        varis = ['X_'+str(mode), 'Y_'+str(mode), 'Time_'+str(mode)]
        for var in varis:
            if var in f.keys():
                del f[var]
                
        f.create_dataset('X_'+str(mode), data=X_clean)
        f.create_dataset('Y_'+str(mode), data=Y_clean)
        f.create_dataset('Time_'+str(mode), data=Time_clean)
    
    return stats
       

def add_circular(df, col):
    rad = np.deg2rad(df[col])
    return np.sin(rad), np.cos(rad)

     
def process_inputs(df, stats=None):
    """
    Preprocess input DataFrame and return feature matrix X.
    If `stats` is provided, it uses the given means and stds.
    Otherwise, computes them and returns for future use.
    
    Returns:
        X: np.ndarray of shape (N, num_features)
        stats: dict containing mean/std for all normalized variables
    """

    for col in ['altitude', 'upstream_nsw', 'f107_mars']:
        df[col] = df[col].astype(np.float64)

    # Sin/cos features
    df['geo_longitude_sin'], df['geo_longitude_cos'] = add_circular(df, 'geo_longitude')
    df['ls_mars_sin'], df['ls_mars_cos'] = add_circular(df, 'ls_mars')
    df['subsolar_lng_sin'], df['subsolar_lng_cos'] = add_circular(df, 'subsolar_lng')
    df['mse_azimuthal_sin'], df['mse_azimuthal_cos'] = add_circular(df, 'mse_azimuthal')

    # Latitude-related
    df['geo_latitude_norm'] = df['geo_latitude'] / 90
    df['subsolar_lat_norm'] = df['subsolar_lat'] / 90
    df['cos_sza'] = np.cos(np.deg2rad(df['mso_sza']))

    # Stack arrays
    imf_array = np.stack(df['upstream_imf'])
    df['imf_x'], df['imf_y'], df['imf_z'] = imf_array[:, 0], imf_array[:, 1], imf_array[:, 2]

    usw_array = np.stack(df['upstream_usw'])
    df['usw_x'], df['usw_y'], df['usw_z'] = usw_array[:, 0], usw_array[:, 1], usw_array[:, 2]

    # Variables to normalize
    norm_cols = [
        'altitude', 'imf_x', 'imf_y', 'imf_z',
        'usw_x', 'usw_y', 'usw_z',
        'upstream_nsw', 'f107_mars'
    ]

    if stats is None:
        stats = {}
        for col in norm_cols:
            mean = df[col].mean()
            std = df[col].std()
            stats[col] = {'mean': mean, 'std': std}
            df[col + '_norm'] = (df[col] - mean) / std
    else:
        for col in norm_cols:
            mean = stats[col]['mean']
            std = stats[col]['std']
            df[col + '_norm'] = (df[col] - mean) / std

    features = [
        'geo_longitude_sin', 'geo_longitude_cos',
        'ls_mars_sin', 'ls_mars_cos',
        'subsolar_lng_sin', 'subsolar_lng_cos',
        'mse_azimuthal_sin', 'mse_azimuthal_cos',
        'geo_latitude_norm', 'subsolar_lat_norm', 'cos_sza',
        'altitude_norm',
        'imf_x_norm', 'imf_y_norm', 'imf_z_norm',
        'usw_x_norm', 'usw_y_norm', 'usw_z_norm',
        'upstream_nsw_norm', 'f107_mars_norm'
    ]
    array_list = []
    for feat in features:
        array_list.append(df[feat])  # should work one-by-one-

    X = np.stack(array_list, axis=-1)

    return X, stats, df


def find_indices_below_mpb(data_dict: Dict[str, np.ndarray], 
                           model_data: Dict[str, Any],
                           mask) -> np.ndarray:
    """
    Identifies array indices for samples below the Mars MPB from a dictionary.

    This function takes a dictionary of NumPy arrays and a pre-loaded empirical model. It
    interpolates the Magnetic Pileup Boundary (MPB) altitude for each sample's
    Solar Zenith Angle (SZA) and returns the indices of samples whose
    altitude is below this boundary.

    Args:
        data_dict (Dict[str, np.ndarray]): Input dictionary of NumPy arrays. Must contain
                                           the keys 'altitude' (in km) and 'sza' (in degrees).
                                           All arrays should have the same length.
        model_data (Dict[str, Any]): A dictionary containing the boundary model,
                                     with keys 'sza_grid' and 'alt_mpb'.

    Returns:
        np.ndarray: A 1D NumPy array of integer indices from the original
                    arrays that correspond to samples below the MPB.

    Raises:
        KeyError: If the data_dict is missing 'altitude' or 'sza' keys,
                  or if the model_data is missing 'sza_grid' or 'alt_mpb'.
    """
    # --- 1. Validate inputs ---
    required_data_keys = ['altitude', 'cos_sza']
    if not all(key in data_dict for key in required_data_keys):
        raise KeyError(f"Input dictionary must contain the keys: {required_data_keys}")

    required_model_keys = ['sza_grid', 'alt_mpb']
    if not all(key in model_data for key in required_model_keys):
        raise KeyError(f"Model data dictionary must contain the keys: {required_model_keys}")

    altitude = data_dict['altitude'][mask]
    sza = np.rad2deg(np.arccos(data_dict['cos_sza'].flatten()))[mask]
    sza_grid = model_data['sza_grid']
    alt_mpb = model_data['alt_mpb']

    # --- 2. Interpolate the MPB altitude for each data point ---
    # For each 'sza' value in the dictionary, find the corresponding
    # boundary altitude from the model's grid using linear interpolation.
    mpb_altitude_at_sza = np.interp(sza, sza_grid, alt_mpb)

    # --- 3. Identify points below the boundary ---
    # This creates a boolean array (True for samples below the MPB)
    is_below_mpb = altitude < mpb_altitude_at_sza

    # --- 4. Get and return the indices ---
    # np.where returns the indices where the condition is True.
    # The [0] is necessary because np.where returns a tuple of arrays.
    below_mpb_indices = np.where(is_below_mpb)[0]

    return below_mpb_indices


########################## ML ####################

class NormalMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pdf_interp = pdf_interp

    def forward(self, y_pred, y_true):

        mse = (y_true - y_pred) ** 2
        weighted_mse = torch.mean(mse, dim=1)
        return weighted_mse.mean()
    

class Mag_Diffweight_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pdf_interp = pdf_interp

    def forward(self, y_pred, y_true):
        
        # st()
        y_real = y_true[:, :3]         
        y_prev = (y_true[:, :3] - y_true[:, 3:])
        
        weights = torch.abs(torch.mean(y_real**2, dim=1) - torch.mean(y_prev**2, dim=1))
        mse = torch.abs(torch.mean(y_real**2, dim=1) - torch.mean(y_pred**2, dim=1))
        
        weighted_mse = torch.mean(mse*weights)
        return weighted_mse
    
    
class Diffweight_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pdf_interp = pdf_interp

    def forward(self, y_pred, y_true):
        
        y_diff = y_true[:, 3:]
        y_real = y_true[:, :3]
        weights = torch.mean(y_diff**2, axis=1)
        mse = (y_real - y_pred) ** 2
        weighted_mse = torch.mean(mse*weights, dim=1)
        return weighted_mse.mean()
    
    
class WeightedMSELoss(nn.Module):
    def __init__(self, pdf_interp):
        super().__init__()
        self.pdf_interp = pdf_interp

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_mag = torch.sqrt(torch.mean(y_true**2, axis=1)).cpu().numpy()
            weights = 1.0 / (self.pdf_interp(y_mag) + 1e-6)
            weights = torch.tensor(weights, 
                                   dtype=torch.float32, 
                                   device=y_true.device)

        mse = (y_true - y_pred) ** 2
        # st()
        weighted_mse = torch.mean(mse, dim=1) * weights
        return weighted_mse.mean()


class MLPRegressor(LightningModule):
    def __init__(self, 
                 input_dim=16, 
                 hidden_dims=[64, 32], 
                 dropout=0.0,
                 lr=1e-3, 
                 weight_decay=1e-4,
                 pdf_interp=None):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))          # normalize features
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))        # regularization
            in_dim = h
        layers.append(nn.Linear(in_dim, 3))  # Output layer

        self.model = nn.Sequential(*layers)
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = WeightedMSELoss(pdf_interp)
        # self.loss_fn = NormalMSELoss()
        # self.loss_fn = Diffweight_MSELoss()
        self.loss_fn = Mag_Diffweight_MSELoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)  # Output shape: [batch_size]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss)

    def configure_optimizers(self):
        # return torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)



def train_gradient_boosting(X, Y, test_size=0.2, random_state=42):
    """
    Train a Gradient Boosting Regressor on X -> Y.
    
    Parameters:
    - X: np.ndarray, shape (N, 20)
    - Y: np.ndarray, shape (N,)
    - test_size: float, proportion of data to use as validation
    - random_state: int, for reproducibility

    Returns:
    - model: trained GradientBoostingRegressor
    - metrics: dict with RMSE and R^2
    """

    # 1. Split data
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # 2. Define model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state
    )

    # 3. Train
    model.fit(X_train, Y_train)

    # 4. Evaluate
    Y_pred = model.predict(X)
    rmse = root_mean_squared_error(Y_val, Y_pred, squared=False)
    r2 = r2_score(Y_val, Y_pred)

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²:   {r2:.4f}")

    return Y_pred


########################## Visualization ####################
    

def plot_dual_binned_prediction_contour(
    sza, altitude, measure, predict,
    idx_sel,
    bins=40,
    label1="Measurments",
    label2="Prediction",
    figname='Figs/sza_vs_alt.png',
    title="Binned Contour Comparison"):
    
    """
    Plot two side-by-side panels with 2D heatmap + contour using two prediction sets.
    
    Parameters:
    - sza: array-like, Solar Zenith Angle
    - altitude: array-like, Altitude
    - measure: array-like, Measurmentsd variable
    - predict: array-like, predicted variable
    - idx_sel: array-like, the indices of sample selected
    - bins: int, number of bins for heatmap
    - label1: title for left panel
    - label2: title for right panel
    - title: overall figure title
    """
    
    # Compute binned means
    stat1, xedges, yedges, _ = binned_statistic_2d(sza[idx_sel], 
                                                   altitude[idx_sel], 
                                                   measure[idx_sel], 
                                                   statistic='mean', 
                                                   bins=bins)
    stat2, _, _, _ = binned_statistic_2d(sza[idx_sel], 
                                         altitude[idx_sel], 
                                         predict[idx_sel], 
                                         statistic='mean', 
                                         bins=bins)

    # Grid centers
    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    ycenters = 0.5 * (yedges[1:] + yedges[:-1])
    X, Y = np.meshgrid(xcenters, ycenters)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(bottom=0.2)  # reserve bottom space for colorbar
    fig.suptitle(title, fontsize=16)
    pcm_list = []

    for ax, stat, label in zip(axes, [stat1, stat2], [label1, label2]):
        pcm = ax.pcolormesh(xedges, yedges, stat.T, cmap='viridis', shading='auto')
        contour = ax.contour(X, Y, stat.T, colors='black', linewidths=1)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('Solar Zenith Angle (deg)')
        ax.set_title(label)
        # fig.colorbar(pcm, ax=ax, label='Mean Prediction')
        ax.grid(True)
        pcm_list.append(pcm)

    axes[0].set_ylabel('Altitude (km)')
    
    # Add a single shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.02])
    cbar = fig.colorbar(pcm_list[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label("B1 (nT)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_lat_lon_B1_dualpanel(
    lat, lon, alt, B1_true, B1_pred, 
    target_alt, alt_tol=10, bins=100,
    figname='Figs/lat_vs_lon.png',
    title="B1 Comparison at Altitude Slice"
):
    """
    Plots true vs. predicted B1 in lat-lon at a specific altitude slice.
    
    Parameters:
    - lat, lon, alt: 1D arrays of positions
    - B1_true, B1_pred: 1D arrays of B1 values (ground truth & predicted)
    - target_alt: altitude level to extract (e.g., 400 km)
    - alt_tol: vertical tolerance for selecting the slice (±km)
    - bins: number of bins for 2D grid
    - title: overall figure title
    """
    # Mask for altitude slice
    mask = np.abs(alt - target_alt) <= alt_tol
    if not np.any(mask):
        raise ValueError("No data found within ±{} km of target altitude: {}".format(alt_tol, target_alt))

    lat_slice = lat[mask]
    lon_slice = lon[mask]
    true_slice = B1_true[mask]
    pred_slice = B1_pred[mask]

    # Compute binned statistics
    stat_true, xedges, yedges, _ = binned_statistic_2d(
        lon_slice, lat_slice, true_slice, statistic='mean', bins=bins
    )
    stat_pred, _, _, _ = binned_statistic_2d(
        lon_slice, lat_slice, pred_slice, statistic='mean', bins=[xedges, yedges]
    )

    # Grid centers
    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    ycenters = 0.5 * (yedges[1:] + yedges[:-1])
    X, Y = np.meshgrid(xcenters, ycenters)

    # Shared color scale
    vmin = np.nanmin([stat_true, stat_pred])
    vmax = np.nanmax([stat_true, stat_pred])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(bottom=0.15)
    fig.suptitle(f"{title} @ {target_alt} ± {alt_tol} km", fontsize=16)

    for ax, data, label in zip(axes, [stat_true, stat_pred], ["Ground Truth", "Prediction"]):
        pcm = ax.pcolormesh(xedges, yedges, data.T, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(label)
        ax.set_xlabel("Longitude (deg)")
        ax.set_aspect("auto")
        ax.grid(True)

    axes[0].set_ylabel("Latitude (deg)")

    # Shared horizontal colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("B1 Value")
    plt.savefig(figname, dpi=300, bbox_inches='tight')

    plt.show()
    

def plot_prediction_comparison(pred, Y, bins=100, figname=None, title='Prediction vs Ground Truth'):
    """
    Plot 2D histogram of predicted vs measured values.

    Parameters:
        pred (np.ndarray): Predicted values, shape (N,)
        Y (np.ndarray): Measured values, shape (N,)
        bins (int): Number of bins for histogram
        figname (str): Path to save the figure. If None, just displays it.
        title (str): Title of the plot
    """
    if pred.shape != Y.shape:
        raise ValueError("pred and Y must have the same shape")

    # Remove NaN/inf
    mask = np.isfinite(pred) & np.isfinite(Y)
    pred_clean = pred[mask]
    Y_clean = Y[mask]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(Y_clean, 'b.', label='Observed |B1|')
    plt.plot(pred_clean, 'y.', label='predicted |B1|')
    # plt.colorbar(label='Counts')
    # plt.plot([Y_clean.min(), Y_clean.max()],
    #          [Y_clean.min(), Y_clean.max()], 'r--', label='y = x')
    plt.xlabel('Date')
    plt.ylabel('|B1| (nT)')
    plt.title(title)
    plt.legend()

    if figname:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        # plt.close()
    # else:
    plt.show()
    
