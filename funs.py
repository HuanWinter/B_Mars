import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d


########################## Basic function ####################


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


def process_inputs(df):
    # 1. Circular features → sin and cos
    def add_circular(df, col):
        rad = np.deg2rad(df[col])
        return np.sin(rad), np.cos(rad)
    
    for col in ['altitude', 'upstream_nsw', 'f107_mars']:
        df[col] = df[col].astype(np.float64)

    # sin/cos for [0, 360] variables
    df['geo_longitude_sin'], df['geo_longitude_cos'] = add_circular(df, 'geo_longitude')
    df['ls_mars_sin'], df['ls_mars_cos'] = add_circular(df, 'ls_mars')
    df['subsolar_lng_sin'], df['subsolar_lng_cos'] = add_circular(df, 'subsolar_lng')
    df['mse_azimuthal_sin'], df['mse_azimuthal_cos'] = add_circular(df, 'mse_azimuthal')

    # 2. Normalize latitude and subsolar_lat from [-90, 90] → [-1, 1]
    df['geo_latitude_norm'] = df['geo_latitude'] / 90
    df['subsolar_lat_norm'] = df['subsolar_lat'] / 90

    # 3. Cosine of solar zenith angle (if in degrees)
    df['cos_sza'] = np.cos(np.deg2rad(df['mso_sza']))

    # 4. Altitude — normalized (optional: log-scale if necessary)
    df['altitude_norm'] = (df['altitude'] - df['altitude'].mean()) / df['altitude'].std()

    # 5. Upstream IMF (Bx, By, Bz) — assume it's an [N, 3] array column
    imf_array = np.stack(df['upstream_imf'].values)
    df['imf_x'] = imf_array[:, 0].astype(np.float64)
    df['imf_y'] = imf_array[:, 1].astype(np.float64)
    df['imf_z'] = imf_array[:, 2].astype(np.float64)

    # 6. Upstream USW (vx, vy, vz)
    usw_array = np.stack(df['upstream_usw'].values)
    df['usw_x'] = usw_array[:, 0].astype(np.float64)
    df['usw_y'] = usw_array[:, 1].astype(np.float64)
    df['usw_z'] = usw_array[:, 2].astype(np.float64)

    # Normalize each component (z-score)
    for col in ['imf_x', 'imf_y', 'imf_z', 'usw_x', 'usw_y', 'usw_z']:
        df[col + '_norm'] = (df[col] - df[col].mean()) / df[col].std()

    # 7. Normalize upstream_nsw and f107_mars
    df['upstream_nsw_norm'] = (df['upstream_nsw'] - df['upstream_nsw'].mean()) / df['upstream_nsw'].std()
    df['f107_mars_norm'] = (df['f107_mars'] - df['f107_mars'].mean()) / df['f107_mars'].std()

    # Combine all features into X
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

    X = df[features].to_numpy(dtype=np.float32)
    return X


########################## ML ####################


class MLPRegressor(LightningModule):
    def __init__(self, input_dim=16, hidden_dims=[64, 32], lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # Output layer

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

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