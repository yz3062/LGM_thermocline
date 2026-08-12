#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-07-12

@author: zhou

Four panels, arranged in two rows (row 1: three columns, row 2: one
enlarged panel spanning the full width)
Top left: climatology with Hol cores
Top middle: multi-model-mean (iCESM, CESM & AWI) piControl with Hol cores
Top right: multi-model-mean LGM with LGM cores
Bottom (enlarged): multi-model-mean LGM − piControl thermocline difference
                    with core z0, plus stippling where all three models agree
                    on the sign of the anomaly

This is based on
Map_core_iCESM_pymc_on_warped_climatology_climatology_diff_vertical.py, with
the iCESM thermocline background in axes[1-3] replaced by the multi-model mean
of iCESM, CESM and AWI.

Core-based thermocline based on pymc d_center_gradient, except the axes[3]
core scatter which uses the posterior median of z0 from
pseudo_d18Ocline_climatology.d18Ocline_pymc_climatology (the depth shift of
the warped-climatology template fit relative to modern climatology), read
directly from Core_d18O_pymc_inference_climatology.
"""

import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import arviz as az
import pseudo_d18Ocline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tr
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in linestrings')
warnings.filterwarnings('ignore', message='invalid value encountered in create_collection')

sns.set(font='Arial',palette='husl',style='ticks',context='paper')

cmap = 'cubehelix_r'

highlight_cores = {'V19-27', 'RC8-102', 'RC11-238', 'RC13-140', 'ML1208-28BB'}

# =============================================================================
# MULTI-MODEL-MEAN CONFIGURATION (iCESM, CESM & AWI)
# =============================================================================

mmm_models = ['iCESM', 'CESM', 'AWI']

# Per-model xlsx files containing d_center_gradient (thermocline depth from
# the PyMC fit).  iCESM LGM is split across two coverages.
xlsx_files = {
    'iCESM': {
        'pi':   'iCESM_pseudo_thermocline_pymc_piControl.xlsx',
        'lgm':  'iCESM_pseudo_thermocline_pymc_lgm.xlsx',
        'lgm2': 'iCESM_pseudo_thermocline_pymc_lgm_2.xlsx',
    },
    'CESM': {
        'pi':  'CESM_pseudo_thermocline_pymc_piControl.xlsx',
        'lgm': 'CESM_pseudo_thermocline_pymc_lgm.xlsx',
    },
    'AWI': {
        'pi':  'AWI_pseudo_thermocline_pymc_piControl_updated_function.xlsx',
        'lgm': 'AWI_pseudo_thermocline_pymc_lgm_updated_function.xlsx',
    },
}

# Common regular grid onto which every model is regridded before averaging
grid_lons = np.arange(120, 300.001, 0.5)
grid_lats = np.arange(-25, 25.001, 0.5)
grid_lon2d, grid_lat2d = np.meshgrid(grid_lons, grid_lats)


def load_thermocline_grid(xlsx_path, xlsx_path2=None):
    """
    Read one or two PyMC thermocline xlsx files, average overlapping (lat, lon)
    cells, and 2-D regrid the scattered d_center_gradient points onto the
    common regular (grid_lon2d, grid_lat2d) grid with scipy.griddata.

    Returns a 2-D array of thermocline depths (m) on the common grid;
    NaN outside each model's coverage.
    """
    df = pd.read_excel(xlsx_path)
    if xlsx_path2 is not None:
        df2 = pd.read_excel(xlsx_path2)
        df  = pd.concat([df, df2], ignore_index=True)

    df = df.groupby(['lat', 'lon'], as_index=False).mean()
    df = df.sort_values(by=['lat', 'lon'])
    df = df.dropna(subset=['lon', 'lat', 'd_center_gradient'])
    df = df[np.isfinite(df['d_center_gradient'])]
    df['lon'] = df['lon'] % 360.0

    return griddata(
        (df['lon'].values, df['lat'].values),
        df['d_center_gradient'].values,
        (grid_lon2d, grid_lat2d),
        method='linear'
    )

#%% set up subplots
# Row 1: three columns (axes[0-2]); Row 2: one enlarged panel spanning
# the full width (axes[3]) — same map aspect ratio, just a bigger box.
fig = plt.figure(figsize=(10, 6.5))
gs = GridSpec(2, 3, figure=fig, hspace=-0.2, wspace=0.08)
proj = ccrs.LambertCylindrical(central_longitude=210)
axes = [
    fig.add_subplot(gs[0, 0], projection=proj),
    fig.add_subplot(gs[0, 1], projection=proj),
    fig.add_subplot(gs[0, 2], projection=proj),
    fig.add_subplot(gs[1, :], projection=proj),
]

#%% Modern climatology
# read in data
df = pd.read_excel('Predicted_d18O_pymc_1000m.xlsx')

z = df['d_center_gradient'].values.astype(float)

# Keep only rows where z is finite
finite = np.isfinite(z)
lon_f = df['lon'].values[finite]
lat_f = df['lat'].values[finite]
z_f   = z[finite]

# Build triangulation from finite points only — no masking needed
triang = tr.Triangulation(lon_f, lat_f)

ct0 = axes[0].tricontourf(
    triang, z_f,
    vmin=0, vmax=320,
    transform=ccrs.LambertCylindrical(),
    cmap=cmap,
    levels=np.linspace(0, 320, 21),
    extend='max')

#%% Multi-model-mean thermocline grids (iCESM, CESM & AWI)

pi_grids  = {}
lgm_grids = {}
for m in mmm_models:
    print(f"Loading {m} thermocline...")
    xf = xlsx_files[m]
    pi_grids[m]  = load_thermocline_grid(xf['pi'])
    lgm_grids[m] = load_thermocline_grid(xf['lgm'], xf.get('lgm2'))

mmm_pi  = np.nanmean([pi_grids[m]  for m in mmm_models], axis=0)
mmm_lgm = np.nanmean([lgm_grids[m] for m in mmm_models], axis=0)

levels_thermo = np.linspace(0, 320, 21)  # 20 intervals between 0 and 320

#%% MMM piControl (axes[1])
ct1 = axes[1].contourf(grid_lons, grid_lats, mmm_pi,
                       levels=levels_thermo,
                       vmin=0, vmax=320,
                       transform=ccrs.LambertCylindrical(),
                       cmap=cmap,
                       extend='max')

#%% MMM LGM (axes[2])
cmap = 'cubehelix_r'
ct2 = axes[2].contourf(grid_lons, grid_lats, mmm_lgm,
                       levels=levels_thermo,
                       vmin=0, vmax=320,
                       transform=ccrs.LambertCylindrical(),
                       cmap=cmap,
                       extend='max')

#%% MMM LGM − piControl difference + sign-agreement stippling (axes[3])

diff = mmm_lgm - mmm_pi

levels_diff = np.linspace(-45, 45, 21)

ct3 = axes[3].contourf(grid_lons, grid_lats, diff,
                 levels=levels_diff,
                 cmap='RdBu',
                 transform=ccrs.LambertCylindrical(),
                 extend='both')

# Stippling: locations where all three models agree on the sign of the
# LGM − piControl thermocline anomaly (and all three have data there)
model_diffs = [lgm_grids[m] - pi_grids[m] for m in mmm_models]
finite_all  = np.all([np.isfinite(d) for d in model_diffs], axis=0)
all_pos     = np.all([d > 0 for d in model_diffs], axis=0)
all_neg     = np.all([d < 0 for d in model_diffs], axis=0)
stipple_mask = finite_all & (all_pos | all_neg)

stipple = axes[3].contourf(grid_lons, grid_lats, stipple_mask.astype(float),
                           levels=[0.5, 1.5], hatches=['....'],
                           colors='none', extend='neither',
                           transform=ccrs.LambertCylindrical(), zorder=1.5)
stipple.set_edgecolor('0.2')
stipple.set_linewidth(0)

#%% get a list of LGM or Hol itrace inference
itrace_files = os.listdir('Core_d18O_pymc_inference_on_warped_climatology')
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

# find core_names with both Hol and LGM files
core_name_both_Hol_LGM = []

for file_Hol in itrace_file_Hol:
    for file_LGM in itrace_file_LGM:
        core_name_Hol = file_Hol.removesuffix('_Hol.nc')
        core_name_LGM = file_LGM.removesuffix('_LGM.nc')
        if core_name_Hol == core_name_LGM:
            core_name_both_Hol_LGM.append(core_name_Hol)

itrace_path = 'Core_d18O_pymc_inference_on_warped_climatology'
# posterior z0 (depth shift relative to modern climatology) lives in the
# warped-climatology-fit inference files, not the erf-refit files above
z0_itrace_path = 'Core_d18O_pymc_inference_climatology'

# get Hol thermocline depth
for itrace_file in itrace_file_Hol:
    try:
        idata_d18Ocline_model_Hol = az.from_netcdf(
            os.path.join(itrace_path, itrace_file)
        )
    except (OSError, ValueError) as e:
        print(f"Skipping {itrace_file}: {e}")
        continue

    # get median thermocline depth
    d_center_gradient = idata_d18Ocline_model_Hol.posterior['d_center_gradient'].median().values
    d18O_min = idata_d18Ocline_model_Hol.posterior['d18O_min'].median().values
    d18O_max = idata_d18Ocline_model_Hol.posterior['d18O_max'].median().values
    thickness = idata_d18Ocline_model_Hol.posterior['thickness'].median().values
    thermocline_std = idata_d18Ocline_model_Hol.posterior['d_center_gradient'].std().values
    tilt = idata_d18Ocline_model_Hol.posterior['tilt'].median().values

    # get predicted d18O profile
    depth, d18O = pseudo_d18Ocline.d18Ocline(d18O_min,
                                             d18O_max,
                                             1000,
                                             d_center_gradient,
                                             thickness,
                                             tilt)

    # get the intersection of predicted d18O profile and -0.66 permil
    # first, create an xarray object
    d18O_xr = xr.DataArray(d18O, coords={'depth': depth})
    thermocline_median = pseudo_d18Ocline.isosurface(d18O_xr, -0.66, dim='depth')

    # get core name
    core_name = itrace_file.removesuffix('_Hol.nc')
    # get core location
    site_lon, site_lat = pseudo_d18Ocline.fetch_location(core_name, include_depth=False)
    # convert from -180 to 180 to 0 to 360
    site_lon_rolled = site_lon % 360
    
    ######### TEST ############
    # if site_lon_rolled > 180 and site_lon_rolled < 240:
    #     print(f'{core_name}: {site_lat}, {site_lon_rolled}, {d_center_gradient}')
    ###########################

    # plot
    axes[0].scatter(site_lon_rolled, site_lat,
                    c=d_center_gradient,
                    s=50,
                    vmin=0, vmax=320,
                    transform=ccrs.LambertCylindrical(),
                    edgecolors='w',
                    cmap=cmap,
                    zorder=6 if core_name in highlight_cores else 3)
    axes[1].scatter(site_lon_rolled, site_lat,
                    c=d_center_gradient,
                    s=50,
                    vmin=0, vmax=320,
                    transform=ccrs.LambertCylindrical(),
                    edgecolors='w',
                    cmap=cmap,
                    zorder=6 if core_name in highlight_cores else 3)

# get LGM thermocline depth
for itrace_file in itrace_file_LGM:
    try:
        idata_d18Ocline_model_LGM = az.from_netcdf(
            os.path.join(itrace_path, itrace_file)
        )
    except (OSError, ValueError) as e:
        print(f"Skipping {itrace_file}: {e}")
        continue

    # get median thermocline depth
    d_center_gradient = idata_d18Ocline_model_LGM.posterior['d_center_gradient'].median().values
    d18O_min = idata_d18Ocline_model_LGM.posterior['d18O_min'].median().values
    d18O_max = idata_d18Ocline_model_LGM.posterior['d18O_max'].median().values
    thickness = idata_d18Ocline_model_LGM.posterior['thickness'].median().values
    thermocline_std = idata_d18Ocline_model_LGM.posterior['d_center_gradient'].std().values
    tilt = idata_d18Ocline_model_LGM.posterior['tilt'].median().values

    # get predicted d18O profile
    depth, d18O = pseudo_d18Ocline.d18Ocline(d18O_min,
                                             d18O_max,
                                             1000,
                                             d_center_gradient,
                                             thickness,
                                             tilt)

    # get the intersection of predicted d18O profile and 0.62 permil
    # Calculation from: LGM_Z20_depth.py
    # first, create an xarray object
    d18O_xr = xr.DataArray(d18O, coords={'depth': depth})
    thermocline_median = pseudo_d18Ocline.isosurface(d18O_xr, 0.62, dim='depth')

    # get core name
    core_name = itrace_file.removesuffix('_LGM.nc')
    # get core location
    site_lon, site_lat = pseudo_d18Ocline.fetch_location(core_name, include_depth=False)
    # convert from -180 to 180 to 0 to 360
    site_lon_rolled = site_lon % 360

    # skip these cores as discussed in the SI
    if core_name == 'V28-213':
        continue

    # plot
    axes[2].scatter(site_lon_rolled, site_lat,
                    c=d_center_gradient,
                    s=50,
                    vmin=0, vmax=320,
                    transform=ccrs.LambertCylindrical(),
                    edgecolors='w',
                    cmap=cmap,
                    zorder=6 if core_name in highlight_cores else 3)

    # depth shift (z0) of the LGM warped-climatology fit relative to modern
    # climatology, from pseudo_d18Ocline_climatology.d18Ocline_pymc_climatology
    try:
        idata_z0_LGM = az.from_netcdf(
            os.path.join(z0_itrace_path, core_name + '_LGM.nc')
        )
    except (OSError, ValueError) as e:
        print(f"Skipping z0 for {core_name}: {e}")
        continue
    z0_median = idata_z0_LGM.posterior['z0'].median().values

    sc = axes[3].scatter(site_lon_rolled, site_lat,
                    c=z0_median,
                    s=50,
                    vmin=-55, vmax=55,
                    transform=ccrs.LambertCylindrical(),
                    edgecolors='k',
                    cmap='RdBu',
                    zorder=6 if core_name in highlight_cores else 3)

#%% beautification

# ICE6G
ds_ice6g = xr.open_dataset('I6_C.VM5a_10min.21.nc', decode_times=False)['Topo'].coarsen(lat=9, lon=9).mean()

LON2D, LAT2D = np.meshgrid(ds_ice6g.lon, ds_ice6g.lat)

axes[0].coastlines()
axes[0].add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
axes[0].gridlines(draw_labels=['left', 'bottom'], color = "None")
axes[0].set_extent([130, 290, -20, 20], crs=ccrs.LambertCylindrical())
axes[0].set_title('Hol profiles', loc='left')
axes[0].set_title('Modern Climatology', loc='right')
axes[0].text(0.02, 0.85, 'a', fontweight='bold', transform=axes[0].transAxes, color='w')

axes[1].coastlines()
axes[1].add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
# drop y-axis (latitude) labels — only bottom (longitude) labels
axes[1].gridlines(draw_labels=['bottom'], color = "None")
axes[1].set_extent([130, 290, -20, 20], crs=ccrs.LambertCylindrical())
axes[1].set_title('Hol profiles', loc='left')
axes[1].set_title('piControl iCESM, CESM2 & AWI', loc='right')
axes[1].text(0.02, 0.85, 'b', fontweight='bold', transform=axes[1].transAxes, color='w')

# Plot land as gray
axes[2].pcolormesh(
    LON2D, LAT2D, np.where(ds_ice6g >= 0, 1.0, np.nan),
    cmap=mcolors.ListedColormap(['#888888']),
    vmin=0, vmax=1,
    transform=ccrs.LambertCylindrical(),
    rasterized=True,
    shading='auto',
    zorder=2,
)
axes[2].coastlines()
# drop y-axis (latitude) labels — only bottom (longitude) labels
axes[2].gridlines(draw_labels=['bottom'], color = "None")
axes[2].set_extent([130, 290, -20, 20], crs=ccrs.LambertCylindrical())
axes[2].set_title('LGM profiles', loc='left')
axes[2].set_title('LGM iCESM, CESM2 & AWI', loc='right')
axes[2].text(0.02, 0.85, 'c', fontweight='bold', transform=axes[2].transAxes, color='w')

# Plot land as gray
axes[3].pcolormesh(
    LON2D, LAT2D, np.where(ds_ice6g >= 0, 1.0, np.nan),
    cmap=mcolors.ListedColormap(['#888888']),
    vmin=0, vmax=1,
    transform=ccrs.LambertCylindrical(),
    rasterized=True,
    shading='auto',
    zorder=2,
)
axes[3].coastlines()
axes[3].gridlines(draw_labels=['left', 'bottom'], color = "None")
axes[3].set_extent([130, 290, -20, 20], crs=ccrs.LambertCylindrical())
axes[3].set_title('LGM profiles - modern climatology', loc='left')
axes[3].set_title('LGM-piControl iCESM, CESM2 & AWI', loc='right')
cbar3 = fig.colorbar(ct3, ax=axes[3], orientation='vertical', aspect=20, pad=0.04)
cbar3.set_label('Δ Thermocline depth (m)')
axes[3].text(0.02, 0.85, 'd', fontweight='bold', transform=axes[3].transAxes, color='w')
# Use MultipleLocator to set ticks at intervals of 15
cbar3.ax.set_yticks([-40, -20, 0, 20, 40])
# flip the colorbar so positive values point down
cbar3.ax.invert_yaxis()

# --- annotation boxes / lines on axes[3] ---
box_kw = dict(transform=ccrs.LambertCylindrical(), color='k', lw=1.5, zorder=4)

# equatorial rectangle: 130E-290E, 6S-10N
axes[3].plot([135, 285, 285, 135, 135], [-6, -6, 10, 10, -6], **box_kw)

# vertical dividers at 190E and 235E (6S-10N)
for xlon in (190, 235):
    axes[3].plot([xlon, xlon], [-6, 10], **box_kw)

# tilted (NW-SE) rectangle: long (SW) edge from (180E, 20S) to (135E, 15N),
# body offset toward the NE so it stays within the map extent
A = np.array([182.0, -14])          # SE end
B = np.array([140.0,  19])          # NW end
perp = np.array([-(B - A)[1], (B - A)[0]])   # perpendicular to A->B
perp = perp / np.hypot(*perp)                # unit vector (points NE here)
width = 9.2                                   # rectangle width (degrees)
C = B + perp * width
D = A + perp * width
axes[3].plot([A[0], B[0], C[0], D[0], A[0]],
             [A[1], B[1], C[1], D[1], A[1]], **box_kw)

# plt.tight_layout(pad=0.5, h_pad=1.6, w_pad=0.3)
plt.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.05)

# Shared horizontal colorbar for panels a-c, placed below their longitude
# labels using the final drawn map positions (aspect-adjusted)
fig.canvas.draw()
p0 = axes[0].get_position()
p2 = axes[2].get_position()
full_w = p2.x1 - p0.x0
cb_w   = 0.65 * full_w
cb_x   = p0.x0 + (full_w - cb_w) / 2
cax = fig.add_axes([cb_x, p0.y0 - 0.075, cb_w, 0.015])
cbar_top = fig.colorbar(ct0, cax=cax, orientation='horizontal')
cbar_top.set_label('Thermocline depth (m)')
cbar_top.ax.set_xticks([0, 80, 160, 240, 320])

# center panel d + its colorbar within the row span, packing the colorbar
# close to the map (the auto-layout leaves a wide gap otherwise)
box_left, box_right = 0.06, 0.90
gap = 0.015                          # gap between map and its colorbar
pd_ax = axes[3].get_position()       # aspect-adjusted (actual drawn) map box
pcb   = cbar3.ax.get_position()
group_w = pd_ax.width + gap + pcb.width
new_map_x0 = box_left + ((box_right - box_left) - group_w) / 2
axes[3].set_position([new_map_x0, pd_ax.y0, pd_ax.width, pd_ax.height])
cbar3.ax.set_position([new_map_x0 + pd_ax.width + gap, pcb.y0, pcb.width, pcb.height])

# label the ends of the (inverted) diverging colorbar
cbar3.ax.text(2, 1.0, 'Shallower LGM\nthermocline',
               transform=cbar3.ax.transAxes, ha='left', va='bottom')
cbar3.ax.text(2, 0.0, 'Deeper LGM\nthermocline',
               transform=cbar3.ax.transAxes, ha='left', va='top')

plt.savefig('Map_core_MMM_pymc_on_warped_climatology_climatology_diff_vertical.png', dpi=700)
# plt.savefig('Map_core_MMM_pymc_on_warped_climatology_climatology_diff_vertical.pdf')
