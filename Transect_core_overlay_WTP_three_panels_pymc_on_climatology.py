#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:22:30 2026

@author: zhou

WTP three panels

left: Hol data on climatology
Middle: LGM data on climatology
Right: LGM data on iCESM
"""

import arviz as az
import os
import matplotlib.pyplot as plt
import pseudo_d18Ocline
import xarray as xr
import numpy as np
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec

sns.set(font='Arial',palette='husl',style='ticks',context='paper')

fig = plt.figure()
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.25)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])

lon_lim = 25
vmin = -2.5
vmax = 2.5
n_levels = 15
cmap = 'coolwarm_r'
# contourf creates smooth, filled contours
# levels=np.linspace ensures the color scale matches your vmin/vmax
levels = np.linspace(vmin, vmax, n_levels)

# read dataset
predicted_d18O_ds = xr.open_dataset('Predicted_d18O.nc', decode_times=False)
# from -180-180 to 0-360
# https://discourse.pangeo.io/t/handling-slicing-with-circular-longitude-coordinates-in-xarray/1608/7
predicted_d18O_ds = predicted_d18O_ds.assign_coords(lon=((360 + (predicted_d18O_ds.lon % 360)) % 360))
predicted_d18O_ds = predicted_d18O_ds.roll(lon=int(len(predicted_d18O_ds['lon']) / 2),roll_coords=True)

# select and average 155-165
transect_ds = predicted_d18O_ds.sel(lon=slice(160-lon_lim, 160+lon_lim+15), lat=slice(-20, 20)).mean(dim='lon')

# plot
cf = axes[0].contourf(transect_ds.lat, transect_ds.depth, transect_ds.d18O_foram, vmin=vmin, vmax=vmax,
             cmap=cmap, levels=levels, extend='both')
axes[1].contourf(transect_ds.lat, transect_ds.depth, transect_ds.d18O_foram, vmin=vmin, vmax=vmax,
             cmap=cmap, levels=levels, extend='both')

#%% extract pymc d_center_gradient that defines thermocline depth

climatology_df = pd.read_excel('Predicted_d18O_pymc_1000m.xlsx')
climatology_df = climatology_df[(climatology_df['lat'] >= -20) &
                                (climatology_df['lat'] <= 20) &
                                (climatology_df['lon'] >= 160-lon_lim) &
                                (climatology_df['lon'] <= 160+lon_lim+15)]

transect_df = climatology_df.groupby('lat', as_index=False).mean()

# plot predicted d18O isosurface -0.66 permil 
axes[0].plot(transect_df.lat, transect_df.d_center_gradient,
             color='k')
axes[1].plot(transect_df.lat, transect_df.d_center_gradient,
             color='k')
#%% Add the colorbar using that mappable
# cbar = fig.colorbar(cf, ax=axes[0], orientation='vertical', pad=0.02, aspect=30)
# cbar.set_label(r'$\mathrm{\delta}^\mathrm{18}$O (‰)') # Using LaTeX for the delta symbol
# cbar.set_ticks(np.arange(-2.5, 2.6, 0.5))
# # Flip the axis so lower values are at the top
# cbar.ax.invert_yaxis()

# cbar = fig.colorbar(cf, ax=axes[1], orientation='vertical', pad=0.02, aspect=30)
# cbar.set_label(r'$\mathrm{\delta}^\mathrm{18}$O (‰)') # Using LaTeX for the delta symbol
# cbar.set_ticks(np.arange(-2.5, 2.6, 0.5))
# # Flip the axis so lower values are at the top
# cbar.ax.invert_yaxis()

#%% column func

def column(d18O_values, lat, depth, vmin, vmax, levels, ax, d_center_gradient, Hol_or_LGM):
    lat_val = lat.item() if hasattr(lat, 'item') else lat
    
    # Define two X-coordinates for the center of the column
    # contourf needs a "width" to interpolate across
    x_coords = np.array([lat_val - 0.25, lat_val + 0.25])
    
    # Broadcast d18O_values to match the x_coords shape
    # This creates a 2D grid where both columns have the same data
    color_data = np.tile(d18O_values[:, np.newaxis], (1, len(x_coords)))
    
    if Hol_or_LGM == 'Hol':
        offset = 0
    elif Hol_or_LGM == 'LGM':
        offset = 1.28
    
    cp = ax.contourf(x_coords, depth, color_data-offset, 
                      levels=levels, 
                      zorder=10, 
                      cmap=cmap, 
                      extend='both',
                      vmin=vmin,
                      vmax=vmax)
    
    # 2. Rectangular Outline
    # Get the boundaries
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = depth.min(), depth.max()
    
    # Draw a rectangle around the whole column
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                         edgecolor='black', facecolor='none', 
                         linewidth=0.2, zorder=12)
    ax.add_patch(rect)
    
    # 3. Add Arrow
    y_arrow = d_center_gradient
    
    ax.annotate('', 
                 xy=(x_max, y_arrow),          # Tip of the arrow (right edge of column)
                 xytext=(x_max + 0.5, y_arrow), # Start of the arrow (pushed to the right)
                 arrowprops=dict(arrowstyle='->', color='black', lw=1),
                 zorder=15)
    
    return cp
    
#%% column

# get a list of LGM or Hol itrace inference
itrace_files = os.listdir('Core_d18O_pymc_inference_on_warped_climatology_constant_ACD')
if '.DS_Store' in itrace_files:
    itrace_files.remove('.DS_Store')

# separate into Hol and LGM
itrace_file_Hol = []
itrace_file_LGM = []
for itrace_file in itrace_files:
    # get core name
    if '_Hol.nc' in itrace_file:
        itrace_file_Hol.append(itrace_file)
    elif '_LGM.nc' in itrace_file:
        itrace_file_LGM.append(itrace_file)

itrace_path = 'Core_d18O_pymc_inference_on_warped_climatology_constant_ACD'

# get Hol thermocline depth
for itrace_file in itrace_file_Hol:
    # get core name
    core_name = itrace_file.removesuffix('_Hol.nc')
    
    # get core location
    site_lon, site_lat, site_depth = pseudo_d18Ocline.fetch_location(core_name, include_depth=True)
    
    # filter for WTP
    if site_lon < (160-lon_lim) or site_lon > (160+lon_lim) or site_lat > 20 or site_lat < -20:
        continue
    
    # filter out sites close to Papua New Guinea
    if site_lon < 150 and site_lon > 0 and site_lat < 0:
        continue
    
    # read itrace
    idata_d18Ocline_model_Hol = az.from_netcdf(os.path.join(itrace_path, itrace_file))
    # get idata parameters
    d_center_gradient = idata_d18Ocline_model_Hol.posterior['d_center_gradient'].median().values
    d18O_min = idata_d18Ocline_model_Hol.posterior['d18O_min'].median().values
    d18O_max = idata_d18Ocline_model_Hol.posterior['d18O_max'].median().values
    tilt = idata_d18Ocline_model_Hol.posterior['tilt'].median().values
    # thickness_fit = idata_depth_thickness.posterior['alpha'].median().values * d_center_gradient**3\
    # - idata_depth_thickness.posterior['beta'].median().values * d_center_gradient**2\
    # + idata_depth_thickness.posterior['gamma'].median().values * d_center_gradient\
    # + idata_depth_thickness.posterior['delta'].median().values
    thickness = idata_d18Ocline_model_Hol.posterior['thickness'].median().values
    thermocline_std = idata_d18Ocline_model_Hol.posterior['d_center_gradient'].std().values
    
    # ############# TEST #############
    # if site_lat < -5 and site_lon > (160-lon_lim) and site_lon < (160+lon_lim):
    #     print(f'{core_name}: {site_lat}, {site_lon}, {d_center_gradient}')
    # print(f'{core_name}: {site_lat}, {d_center_gradient}')
    # ################################
    
    # get predicted d18O profile
    depth, d18O = pseudo_d18Ocline.d18Ocline(d18O_min,
                                             d18O_max,
                                             1000,
                                             d_center_gradient,
                                             thickness,
                                             tilt)
        
    column(d18O, site_lat, depth, vmin, vmax, levels, axes[0], d_center_gradient, 'Hol')

# get LGM thermocline depth
for itrace_file in itrace_file_LGM:
    # get core name
    core_name = itrace_file.removesuffix('_LGM.nc')
    
    # skip these cores as discussed in the SI
    if core_name == 'V28-213':
        continue
    
    # get core location
    site_lon, site_lat, site_depth = pseudo_d18Ocline.fetch_location(core_name, include_depth=True)
    
    # filter for WTP
    if site_lon < (160-lon_lim) or site_lon > (160+lon_lim) or site_lat > 20 or site_lat < -20:
        continue
    
    # read itrace
    idata_d18Ocline_model_LGM = az.from_netcdf(os.path.join(itrace_path, itrace_file))
    # get idata parameters
    d_center_gradient = idata_d18Ocline_model_LGM.posterior['d_center_gradient'].median().values
    d18O_min = idata_d18Ocline_model_LGM.posterior['d18O_min'].median().values
    d18O_max = idata_d18Ocline_model_LGM.posterior['d18O_max'].median().values
    tilt = idata_d18Ocline_model_LGM.posterior['tilt'].median().values
    # thickness_fit = idata_depth_thickness.posterior['alpha'].median().values * d_center_gradient**3\
    # - idata_depth_thickness.posterior['beta'].median().values * d_center_gradient**2\
    # + idata_depth_thickness.posterior['gamma'].median().values * d_center_gradient\
    # + idata_depth_thickness.posterior['delta'].median().values
    thickness = idata_d18Ocline_model_LGM.posterior['thickness'].median().values
    thermocline_std = idata_d18Ocline_model_LGM.posterior['d_center_gradient'].std().values
    
    # get predicted d18O profile
    depth, d18O = pseudo_d18Ocline.d18Ocline(d18O_min,
                                             d18O_max,
                                             1000,
                                             d_center_gradient,
                                             thickness,
                                             tilt)
        
    column(d18O, site_lat, depth, vmin, vmax, levels, axes[1], d_center_gradient, 'LGM')
    column(d18O, site_lat, depth, vmin, vmax, levels, axes[2], d_center_gradient, 'LGM')

#%% iCESM
lon_lim = 25
vmin = -2.5
vmax = 2.5
cmap = 'coolwarm_r'

# read dataset
predicted_d18O_lgm_ds = xr.open_dataset('Predicted_d18O_icesm_Zhu2017_lgm.nc', decode_times=False)
# # from -180-180 to 0-360
# # https://discourse.pangeo.io/t/handling-slicing-with-circular-longitude-coordinates-in-xarray/1608/7
# predicted_d18O_ds = predicted_d18O_ds.assign_coords(lon=((360 + (predicted_d18O_ds.lon % 360)) % 360))
# predicted_d18O_ds = predicted_d18O_ds.roll(lon=int(len(predicted_d18O_ds['lon']) / 2),roll_coords=True)

# Define your limits
lon_min, lon_max = 160 - lon_lim, 160 + lon_lim
lat_min, lat_max = -20, 20

# 1. Slice the data
# We keep a temporary dataset 'mask' to preserve coordinates
mask = predicted_d18O_lgm_ds.where(
    (predicted_d18O_lgm_ds.TLONG >= lon_min) & 
    (predicted_d18O_lgm_ds.TLONG <= lon_max) & 
    (predicted_d18O_lgm_ds.TLAT >= lat_min) & 
    (predicted_d18O_lgm_ds.TLAT <= lat_max), 
    drop=True
)

# 2. Extract 1D latitude array (take the mean across nlon or just the first slice)
# This gives us a 1D array of length 139 to match your nlat dimension
lat_coords = mask.TLAT.mean(dim='nlon')

# 3. Collapse the data variable
transect_ds = mask.mean(dim='nlon')

# Now plot using those 1D coordinates
cf = axes[2].contourf(lat_coords, transect_ds.z_t / 100, transect_ds.d18O_foram,
                    levels=np.linspace(vmin, vmax, 20), cmap=cmap, extend='both')

#%% extract pymc d_center_gradient that defines thermocline depth
# Have to regrid onto a regular grid because otherwise lon average doesn't work

# 1. Read the 2 files each having different coverages
df1_CESM_lgm = pd.read_excel('iCESM_pseudo_thermocline_pymc_lgm.xlsx')
df2_CESM_lgm = pd.read_excel('iCESM_pseudo_thermocline_pymc_lgm_2.xlsx')

# concat
df_CESM_lgm = pd.concat([df1_CESM_lgm, df2_CESM_lgm])

# group by
df_CESM_lgm = df_CESM_lgm.groupby(['lat', 'lon'], as_index=False).mean()

# Ensure coordinates are monotonic (increasing)
df_CESM_lgm = df_CESM_lgm.sort_values(by=['lat', 'lon'])

# 2. Drop rows with non-finite values in any relevant column
df_CESM_lgm = df_CESM_lgm.dropna(subset=['lon', 'lat', 'd_center_gradient'])
df_CESM_lgm = df_CESM_lgm[np.isfinite(df_CESM_lgm['d_center_gradient'])]

# --- Create irregular DataArray from df_CESM_lgm ---
lats = df_CESM_lgm['lat'].values
lons = df_CESM_lgm['lon'].values
values = df_CESM_lgm['d_center_gradient'].values

# --- Define regular 1-degree grid ---
lat_reg = np.arange(np.floor(lats.min()), np.ceil(lats.max()) + 1, 1.0)
lon_reg = np.arange(np.floor(lons.min()), np.ceil(lons.max()) + 1, 1.0)
lon_grid, lat_grid = np.meshgrid(lon_reg, lat_reg)

# --- Regrid using linear interpolation (scattered → regular grid) ---
values_reg = griddata(
    points=(lons, lats),      # irregular source points
    values=values,
    xi=(lon_grid, lat_grid),  # regular target grid
    method='linear'           # or 'nearest' / 'cubic'
)

# --- Build xarray DataArray on the regular grid ---
da_CESM_lgm = xr.DataArray(
    data=values_reg,
    dims=['lat', 'lon'],
    coords={
        'lat': lat_reg,
        'lon': lon_reg,
    },
    name='d_center_gradient',
    attrs={
        'long_name': 'Center Gradient',
        'description': 'Regridded from iCESM LGM pseudo-thermocline data',
        'regrid_method': 'linear interpolation via scipy.griddata',
        'grid_resolution': '1 degree'
    }
)

da_CESM_lgm_WTP = da_CESM_lgm.sel(lon=slice(160-lon_lim, 160+lon_lim))

da_CESM_lgm_mean_lon = da_CESM_lgm_WTP.mean(dim='lon', skipna=True)

# plot predicted d18O isosurface 0.62 permil 
axes[2].plot(da_CESM_lgm_mean_lon.lat, da_CESM_lgm_mean_lon,
             color='k')

#%% Add the colorbar using that mappable
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', aspect=20, pad=0.04)
cbar.set_label(r'$\mathrm{\delta}^\mathrm{18}$O (‰)') # Using LaTeX for the delta symbol
cbar.set_ticks(np.arange(-2.5, 2.6, 0.5))
# Flip the axis so lower values are at the top
cbar.ax.invert_yaxis()

#%% beautification
for i in range(3):
    axes[i].set_ylim(0, 350)
    axes[i].invert_yaxis()
    axes[i].set_xlim(-20, 20)
    # Formatter function to add N/S labels
    def lat_formatter(x, pos):
        # Convert to integer to remove decimal points
        val = int(round(x))
        if x > 0:
            return f"{val}°N"
        elif x < 0:
            return f"{abs(val)}°S"
        else:
            return "0°"
    
    axes[i].xaxis.set_major_formatter(FuncFormatter(lat_formatter))

axes[0].set_ylabel('Depth (m)')
axes[0].set_title('Hol profiles', loc='left')
axes[0].set_title('Modern Climatology', loc='right')
axes[0].text(0.02, 0.95, 'a', fontweight='bold', transform=axes[0].transAxes)

axes[1].set_title('LGM profiles', loc='left')
axes[1].set_title('Modern Climatology', loc='right')
axes[1].text(0.02, 0.95, 'b', fontweight='bold', transform=axes[1].transAxes)

axes[2].set_title('LGM profiles', loc='left')
axes[2].set_title('LGM iCESM', loc='right')
axes[2].text(0.02, 0.95, 'c', fontweight='bold', transform=axes[2].transAxes)

fig.set_size_inches(14, 3.5)

# [left, bottom, width, height] in figure coordinates
pos3 = axes[2].get_position()
cax.set_position([pos3.x1 + 0.01, pos3.y0, 0.015, pos3.height])
plt.subplots_adjust(left=0.06, right=0.90, top=0.9, bottom=0.1)
plt.savefig('Transect_core_overlay_WTP_three_panels_pymc_on_climatology_constant_ACD.png', dpi=700)