#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 00:39:01 2025

@author: zhou

Plot transect of climatology as background and columns of core profiles
(panels a, b), and add the multi-model-mean equatorial dT/dz anomaly panels
(panels c, d) on the third row.

This is based on Transect_core_regional_mean_overlay_gradient_pymc_on_climatology.py
with axes[2] (the single iCESM LGM panel) removed and replaced by the two
multi-model-mean panels from Model_equatorial_thermocline_MMM.py.
"""

import arviz as az
import os
import matplotlib.pyplot as plt
import pseudo_d18Ocline
import xarray as xr
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# MULTI-MODEL-MEAN CONFIGURATION (from Model_equatorial_thermocline_MMM.py)
# =============================================================================

# Define target common grid for interpolation (0 to 360 Lon, 0 to 500m Depth)
target_lon = np.arange(0.5, 360, 2.0)
target_depth = np.arange(0, 505, 5)  # 5m resolution for accurate derivative
LON_GRID, DEPTH_GRID = np.meshgrid(target_lon, target_depth)

# Latitudinal band for both the dT/dz transect and the xlsx thermocline average
LAT_MIN = -6   # 6°S
LAT_MAX = 10   # 10°N

# Model NetCDF file paths and variable names
models = {
    'MPI': {
        'pi':           'ds_MPI_piControl_time_concat.nc',
        'lgm':          'ds_MPI_lgm_time_concat.nc',
        'var':          'thetao',
        'lat':          'latitude',
        'lon':          'longitude',
        'depth':        'lev',
        'depth_factor': 1.0
    },
    'MIROC': {
        'pi':           'ds_MIROC_piControl_time_concat.nc',
        'lgm':          'ds_MIROC_lgm_time_concat.nc',
        'var':          'thetao',
        'lat':          'latitude',
        'lon':          'longitude',
        'depth':        'lev',
        'depth_factor': 1.0
    },
    'AWI': {
        'pi':           'ds_AWI_piControl_time_concat.nc',
        'lgm':          'ds_AWI_lgm_time_concat.nc',
        'var':          'thetao',
        'lat':          'lat',
        'lon':          'lon',
        'depth':        'depth',
        'depth_factor': 1.0
    },
    'CESM': {
        'pi':           'ds_CESM_piControl_time_concat.nc',
        'lgm':          'ds_CESM_lgm_time_concat.nc',
        'var':          'TEMP',
        'lat':          'TLAT',
        'lon':          'TLONG',
        'depth':        'z_t',
        'depth_factor': 100.0  # cm → m
    },
    'iCESM': {
        'pi':           'ds_iCESM_piControl_time_concat.nc',
        'lgm':          'ds_iCESM_lgm_time_concat.nc',
        'var':          'TEMP',
        'lat':          'TLAT',
        'lon':          'TLONG',
        'depth':        'z_t',
        'depth_factor': 100.0  # cm → m
    }
}

# Per-model xlsx files containing d_center_gradient (thermocline depth from
# PyMC Gaussian fit).  Keys must match the 'models' dict above.
xlsx_files = {
    'iCESM': {
        'pi':   'iCESM_pseudo_thermocline_pymc_piControl.xlsx',
        'lgm':  'iCESM_pseudo_thermocline_pymc_lgm.xlsx',
        'lgm2': 'iCESM_pseudo_thermocline_pymc_lgm_2.xlsx',
    },
    'MIROC': {
        'pi':  'MIROC_pseudo_thermocline_pymc_piControl_updated_function.xlsx',
        'lgm': 'MIROC_pseudo_thermocline_pymc_lgm_updated_function.xlsx',
    },
    'MPI': {
        'pi':  'MPI_pseudo_thermocline_pymc_piControl_updated_function.xlsx',
        'lgm': 'MPI_pseudo_thermocline_pymc_lgm_updated_function.xlsx',
    },
    'AWI': {
        'pi':  'AWI_pseudo_thermocline_pymc_piControl_updated_function.xlsx',
        'lgm': 'AWI_pseudo_thermocline_pymc_lgm_updated_function.xlsx',
    },
    'CESM': {
        'pi':  'CESM_pseudo_thermocline_pymc_piControl.xlsx',
        'lgm': 'CESM_pseudo_thermocline_pymc_lgm.xlsx',
    }
}

# Two panels: which models are averaged together in each
panel_groups = [
    ('iCESM, CESM2 & AWI', ['iCESM', 'CESM', 'AWI']),
    ('MPI & MIROC',        ['MPI', 'MIROC']),
]


# =============================================================================
# MULTI-MODEL-MEAN HELPER FUNCTIONS
# =============================================================================

def lon_formatter(x, pos):
    x = ((x + 180) % 360) - 180  # normalise to -180 to 180
    if x > 0:
        return f'{x:.0f}°E'
    elif x < 0:
        return f'{abs(x):.0f}°W'
    else:
        return '0°'


def extract_and_grid_transect(ds, model_info):
    """
    Extract the equatorial temperature slice from a model dataset,
    interpolate to the common (lon, depth) grid, and compute dT/dz.
    """
    T    = ds[model_info['var']].squeeze().values
    lat  = ds[model_info['lat']].values
    lon  = ds[model_info['lon']].values % 360.0
    depth = ds[model_info['depth']].values / model_info['depth_factor']

    # Broadcast depth / lat / lon to match shape of T
    if model_info['lat'] == 'lat' and T.ndim == 2:
        # AWI: unstructured (depth, ncells)
        LON   = lon[np.newaxis, :]   * np.ones_like(T)
        LAT   = lat[np.newaxis, :]   * np.ones_like(T)
        DEPTH = depth[:, np.newaxis] * np.ones_like(T)
    else:
        # Structured grids: (depth, y, x)
        DEPTH = depth[:, np.newaxis, np.newaxis] * np.ones_like(T)
        LAT   = lat[np.newaxis, :, :]            * np.ones_like(T)
        LON   = lon[np.newaxis, :, :]            * np.ones_like(T)

    # Keep only the equatorial band 6°S – 10°N
    eq_mask  = (LAT >= LAT_MIN) & (LAT <= LAT_MAX) & ~np.isnan(T)
    flat_lon   = LON[eq_mask]
    flat_depth = DEPTH[eq_mask]
    flat_T     = T[eq_mask]

    # Scatter-interpolate to the regular (lon × depth) grid
    points     = np.column_stack((flat_lon, flat_depth))
    T_gridded  = griddata(points, flat_T, (LON_GRID, DEPTH_GRID), method='linear')

    # Fill residual horizontal NaN gaps
    for i in range(len(target_depth)):
        row  = T_gridded[i, :]
        mask = np.isnan(row)
        if mask.any() and not mask.all():
            T_gridded[i, mask] = np.interp(
                target_lon[mask], target_lon[~mask], row[~mask]
            )

    dT_dz = np.gradient(T_gridded, target_depth, axis=0)
    return dT_dz


def load_thermocline_from_xlsx(xlsx_path, xlsx_path2=None,
                               lat_min=LAT_MIN, lat_max=LAT_MAX):
    """
    Read one or two PyMC thermocline xlsx files, 2-D regrid the scattered
    d_center_gradient points onto a regular 1° (lat, lon) grid with
    scipy.griddata, then average over the equatorial band [lat_min, lat_max]
    and interpolate to target_lon.

    Returns a 1-D array of thermocline depths (m) on target_lon.
    NaN where no data are available.
    """
    df = pd.read_excel(xlsx_path)

    if xlsx_path2 is not None:
        df2 = pd.read_excel(xlsx_path2)
        df  = pd.concat([df, df2], ignore_index=True)

    # Average overlapping (lat, lon) cells, then restore monotonic order
    df = df.groupby(['lat', 'lon'], as_index=False).mean()
    df = df.sort_values(by=['lat', 'lon'])

    # Drop rows with non-finite values in any relevant column
    df = df.dropna(subset=['lon', 'lat', 'd_center_gradient'])
    df = df[np.isfinite(df['d_center_gradient'])]

    # Normalise longitude to 0–360 (matches target_lon convention)
    df['lon'] = df['lon'] % 360.0

    lats   = df['lat'].values
    lons   = df['lon'].values
    values = df['d_center_gradient'].values

    # 2-D regrid the scattered points onto a regular 1° (lat, lon) grid
    lat_reg = np.arange(np.floor(lats.min()), np.ceil(lats.max()) + 1, 1.0)
    lon_reg = np.arange(np.floor(lons.min()), np.ceil(lons.max()) + 1, 1.0)
    lon_grid, lat_grid = np.meshgrid(lon_reg, lat_reg)

    values_reg = griddata(
        points=(lons, lats),
        values=values,
        xi=(lon_grid, lat_grid),
        method='linear'
    )

    # Average over the equatorial band, then interpolate onto target_lon
    band_mask = (lat_reg >= lat_min) & (lat_reg <= lat_max)
    band_mean = np.nanmean(values_reg[band_mask, :], axis=0)

    valid = np.isfinite(band_mean)
    thermocline = np.interp(
        target_lon,
        lon_reg[valid],
        band_mean[valid],
        left=np.nan,
        right=np.nan
    )
    return thermocline


if __name__ == '__main__':
    # Every size rcParam is set explicitly rather than relying on the relative
    # defaults ('large', 'medium', ...), which some runtimes (e.g. Spyder's
    # inline backend) pin to numbers so that changing font.size alone has no
    # effect on ticks, axis labels and titles.
    plt.rcParams.update({
        'font.size':        7,   # ax.text panel letters, anything unspecified
        'axes.titlesize':   7,
        'axes.labelsize':   7,
        'xtick.labelsize':  6,
        'ytick.labelsize':  6,
        'legend.fontsize':  6,
        'legend.title_fontsize': 6,
        'figure.titlesize': 7,
    })

    lon_lim = 10
    vmin = -2.2
    vmax = 2.2
    n_levels = 9
    cmap = 'coolwarm_r'
    lat_S_lim = -6
    lat_N_lim = 10

    # ── Figure layout: 3 rows.  Rows 0 & 1 span the full width (panels a, b);
    #    row 2 is split into left/right panels (c, d) for the MMM groups. ──
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.05])
    ax0     = fig.add_subplot(gs[0, 0:2])
    ax1     = fig.add_subplot(gs[1, 0:2])
    ax_l    = fig.add_subplot(gs[2, 0])
    ax_r    = fig.add_subplot(gs[2, 1])
    cax_top = fig.add_subplot(gs[0:2, 2])   # colorbar column, rows 0–1 (panels a,b)
    cax_bot = fig.add_subplot(gs[2, 2])     # colorbar column, row 2  (panels c,d)
    axes    = [ax0, ax1]

    # contourf creates smooth, filled contours
    # levels=np.linspace ensures the color scale matches your vmin/vmax
    levels = np.linspace(vmin, vmax, n_levels)

    # read dataset
    predicted_d18O_ds = xr.open_dataset('Predicted_d18O.nc', decode_times=False)
    # from -180-180 to 0-360
    # https://discourse.pangeo.io/t/handling-slicing-with-circular-longitude-coordinates-in-xarray/1608/7
    predicted_d18O_ds = predicted_d18O_ds.assign_coords(lon=((360 + (predicted_d18O_ds.lon % 360)) % 360))
    predicted_d18O_ds = predicted_d18O_ds.roll(lon=int(len(predicted_d18O_ds['lon']) / 2),roll_coords=True)

    # select and average 5S to 5N
    transect_ds = predicted_d18O_ds.sel(lon=slice(150, 281), lat=slice(lat_S_lim, lat_N_lim)).mean(dim='lat')

    # plot
    cf = axes[0].contourf(transect_ds.lon, transect_ds.depth, transect_ds.d18O_foram, vmin=vmin, vmax=vmax,
                 cmap=cmap, levels=levels, extend='both')
    axes[1].contourf(transect_ds.lon, transect_ds.depth, transect_ds.d18O_foram, vmin=vmin, vmax=vmax,
                 cmap=cmap, levels=levels, extend='both')

    #%% extract pymc d_center_gradient that defines thermocline depth

    climatology_df = pd.read_excel('Predicted_d18O_pymc_1000m.xlsx')
    climatology_df = climatology_df[(climatology_df['lat'] >= lat_S_lim) &
                                    (climatology_df['lat'] <= lat_N_lim) &
                                    (climatology_df['lon'] >= 150) &
                                    (climatology_df['lon'] <= 281)]

    transect_df = climatology_df.groupby('lon', as_index=False).mean()

    # plot predicted d18O isosurface -0.66 permil
    axes[0].plot(transect_df.lon, transect_df.d_center_gradient,
                 color='k')
    axes[1].plot(transect_df.lon, transect_df.d_center_gradient,
                 color='k')

    #%% Add the colorbar using that mappable
    # Pass the two d18O axes to the ax parameter so the colorbar is
    # placed to the right of the top two panels.
    cbar = fig.colorbar(cf, cax=cax_top)

    cbar.set_label(r'$\mathrm{\delta}^\mathrm{18}$O (‰)')
    cbar.ax.invert_yaxis()

    #%% beautification

    # Formatter function to add N/S labels
    def lat_formatter(x, pos):
        # Convert to integer to remove decimal points
        val = int(round(x))
        if x > 180:
            return f"{((val + 180) % 360 - 180) * -1}°W"
        elif x < 180:
            return f"{abs(val)}°E"
        else:
            return "0°"

    axes[0].set_ylim(0, 225)
    axes[0].set_xlim(150, 281)
    axes[0].invert_yaxis()
    axes[0].set_xticklabels([])
    axes[0].text(0.02, 0.9, 'a', fontweight='bold', transform=axes[0].transAxes)

    axes[1].set_ylim(0, 225)
    axes[1].set_xlim(150, 281)
    axes[1].invert_yaxis()
    axes[1].set_ylabel('Depth (m)')
    axes[1].xaxis.set_major_formatter(FuncFormatter(lat_formatter))
    axes[1].text(0.02, 0.9, 'b', fontweight='bold', transform=axes[1].transAxes)

    axes[0].set_title('Hol profiles', loc='left')
    axes[0].set_title('Modern climatology', loc='right')

    axes[1].set_title('LGM profiles', loc='left')
    axes[1].set_title('Modern climatology', loc='right')

    #%% arrow helper

    # Arrows are sized on the page rather than in data units, so that a small
    # thermocline offset still shows a stem instead of collapsing into a bare
    # head.  Each arrow returns a closure that re-does the sizing; they are
    # called again once the figure has its final geometry (see before savefig).
    arrow_resizers = []

    def offset_arrow(ax, x, y_tail, y_tip, lw=0.8,
                     head_frac=1.5, max_head_pt=7, min_len_pt=4.0):
        """
        Vertical arrow from y_tail (climatology) to y_tip (core thermocline),
        drawn so a stem is always visible:

        * the head is capped at `head_frac` of the arrow's length on the page
          (and at `max_head_pt`), so it never swallows a short arrow;
        * arrows shorter than `min_len_pt` are lengthened to that minimum by
          extending the *tail*, so the tip stays on the true core thermocline
          and only the direction, not the magnitude, is read off the short ones.

        Returns a callable that redoes the sizing from the current axes
        transform (needed after the figure gets its final size).
        """
        arrow = FancyArrowPatch((x, y_tail), (x, y_tip),
                                arrowstyle='->,head_length=1,head_width=0.5',
                                mutation_scale=max_head_pt,
                                shrinkA=0, shrinkB=0,
                                color='black', lw=lw, zorder=15)
        ax.add_patch(arrow)

        def resize():
            # points per data unit along y, from the current axes transform
            (_, p0), (_, p1) = ax.transData.transform([(x, y_tip), (x, y_tip + 1)])
            pt_per_m = abs(p1 - p0) * 72.0 / ax.figure.dpi
            if pt_per_m == 0:
                return

            length_pt = abs(y_tail - y_tip) * pt_per_m
            y_tail_drawn = y_tail
            if length_pt < min_len_pt:
                # point away from the tip, in the direction of the true tail
                sign = 1.0 if y_tail >= y_tip else -1.0
                y_tail_drawn = y_tip + sign * min_len_pt / pt_per_m
                length_pt = min_len_pt

            arrow.set_positions((x, y_tail_drawn), (x, y_tip))
            arrow.set_mutation_scale(min(max_head_pt, head_frac * length_pt))

        resize()
        return resize

    #%% column func

    def column(d18O_values, lat, depth, vmin, vmax, levels, ax, d_center_gradient,
               d_center_gradient_std, Hol_or_LGM, d_center_gradient_clim):
        """
        d_center_gradient      : core-based thermocline depth
        d_center_gradient_clim : climatology/model thermocline depth at this longitude
        """
        lat_val = lat.item() if hasattr(lat, 'item') else lat

        # Define two X-coordinates for the center of the column
        # contourf needs a "width" to interpolate across
        x_coords = np.array([lat_val - 1, lat_val + 1])

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
        x_min, x_max = x_coords[0], x_coords[-1]
        y_min, y_max = depth.min(), depth.max()

        # Draw a rectangle around the whole column
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             edgecolor='black', facecolor='none',
                             linewidth=0.2, zorder=12)
        ax.add_patch(rect)

        # 3. Short dashed horizontal line marking the core-based thermocline depth
        dash_half_width = 1.5  # half-length of the dashed line in longitude units
        ax.plot([x_min, x_max + dash_half_width*2],
                [d_center_gradient, d_center_gradient],
                color='black', linestyle='--', linewidth=0.8, zorder=15)

        # 3b. Vertical error bar on the dashed line, centred on d_center_gradient
        x_center = (x_min + x_max) / 2          # horizontal centre of the column
        # cap_width = 0.6                          # half-width of the error bar caps
        # ax.errorbar(x_center, d_center_gradient,
        #             yerr=d_center_gradient_std,
        #             fmt='none',
        #             ecolor='black',
        #             elinewidth=1.0,
        #             capsize=3,
        #             capthick=1.0,
        #             zorder=16)

        # 4. Vertical arrow from climatology/model thermocline to core thermocline
        #    Placed just to the right of the dashed line
        x_arrow = x_max + dash_half_width * 2 + 0.5
        arrow_resizers.append(
            offset_arrow(ax, x_arrow,
                         d_center_gradient_clim,  # tail: clim thermocline
                         d_center_gradient)       # tip:  core thermocline
        )

        return cp

    #%% column

    # get a list of LGM or Hol itrace inference
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

    itrace_path = 'Core_d18O_pymc_inference_on_warped_climatology'

    # column longitudes
    col_lons = {'WTP': 158, 'CTP': 200, 'ETP': 260}

    def col_lon_180(lon_0_360):
        """col_lons are in 0-360 convention; is_eastern_pacific expects -180..180."""
        return lon_0_360 - 360 if lon_0_360 > 180 else lon_0_360

    # create arrays of size (1000,) to store the sum of d18O of the three regions
    d18O_WTP = np.zeros(1000)
    d18O_CTP = np.zeros(1000)
    d18O_ETP = np.zeros(1000)
    d18O_WTP_count = 0
    d18O_CTP_count = 0
    d18O_ETP_count = 0
    d_center_gradient_WTP = 0
    d_center_gradient_CTP = 0
    d_center_gradient_ETP = 0
    d_center_gradient_sum_sqr_WTP = 0
    d_center_gradient_sum_sqr_CTP = 0
    d_center_gradient_sum_sqr_ETP = 0

    # get LGM thermocline depth
    for itrace_file in itrace_file_LGM:
        # get core name
        core_name = itrace_file.removesuffix('_LGM.nc')

        # get core location
        site_lon, site_lat, site_depth = pseudo_d18Ocline.fetch_location(core_name, include_depth=True)

        # read itrace
        idata_d18Ocline_model_LGM = az.from_netcdf(os.path.join(itrace_path, itrace_file))
        # get idata parameters
        d_center_gradient = idata_d18Ocline_model_LGM.posterior['d_center_gradient'].median().values
        d18O_min = idata_d18Ocline_model_LGM.posterior['d18O_min'].median().values
        d18O_max = idata_d18Ocline_model_LGM.posterior['d18O_max'].median().values
        tilt = idata_d18Ocline_model_LGM.posterior['tilt'].median().values
        thickness = idata_d18Ocline_model_LGM.posterior['thickness'].median().values
        thermocline_std = idata_d18Ocline_model_LGM.posterior['d_center_gradient'].std().values

        # get predicted d18O profile
        depth, d18O = pseudo_d18Ocline.d18Ocline(d18O_min,
                                                 d18O_max,
                                                 1000,
                                                 d_center_gradient,
                                                 thickness,
                                                 tilt)

        # filter for WTP
        if site_lon > (160-25) and site_lon < (160+25) and site_lat > lat_S_lim and site_lat < lat_N_lim:
            d18O_WTP += d18O
            d18O_WTP_count += 1
            d_center_gradient_WTP += d_center_gradient
        elif site_lon > (-160-10) and site_lon < (-160+10) and site_lat > lat_S_lim and site_lat < lat_N_lim:
            d18O_CTP += d18O
            d18O_CTP_count += 1
            d_center_gradient_CTP += d_center_gradient
        elif site_lon > (-95-25) and site_lon < (-95+25) and site_lat > lat_S_lim and site_lat < lat_N_lim:
            d18O_ETP += d18O
            d18O_ETP_count += 1
            d_center_gradient_ETP += d_center_gradient

    # get d18O profile means
    d18O_WTP_avg = d18O_WTP / d18O_WTP_count
    d18O_CTP_avg = d18O_CTP / d18O_CTP_count
    d18O_ETP_avg = d18O_ETP / d18O_ETP_count

    # --- Interpolate climatology thermocline depths at column longitudes ---
    # axes[1]: Holocene climatology (transect_df)
    clim_LGM_WTP  = float(np.interp(col_lons['WTP'], transect_df['lon'], transect_df['d_center_gradient']))
    clim_LGM_CTP  = float(np.interp(col_lons['CTP'], transect_df['lon'], transect_df['d_center_gradient']))
    clim_LGM_ETP  = float(np.interp(col_lons['ETP'], transect_df['lon'], transect_df['d_center_gradient']))

    # pymc
    idata_WTP_LGM = pseudo_d18Ocline.d18Ocline_pymc_simple(depth, d18O_WTP_avg)
    WTP_posterior_d_center_gradient_LGM = idata_WTP_LGM.posterior['d_center_gradient'].median().values
    WTP_posterior_d_center_gradient_std_LGM = idata_WTP_LGM.posterior['d_center_gradient'].std().values

    idata_CTP_LGM = pseudo_d18Ocline.d18Ocline_pymc_simple(depth, d18O_CTP_avg)
    CTP_posterior_d_center_gradient_LGM = idata_CTP_LGM.posterior['d_center_gradient'].median().values
    CTP_posterior_d_center_gradient_std_LGM = idata_CTP_LGM.posterior['d_center_gradient'].std().values

    idata_ETP_LGM = pseudo_d18Ocline.d18Ocline_pymc_simple(depth, d18O_ETP_avg)
    ETP_posterior_d_center_gradient_LGM = idata_ETP_LGM.posterior['d_center_gradient'].median().values
    ETP_posterior_d_center_gradient_std_LGM = idata_ETP_LGM.posterior['d_center_gradient'].std().values

    # plot the three columns
    column(d18O_WTP_avg, col_lons['WTP'], depth, vmin, vmax, levels, axes[1],
           WTP_posterior_d_center_gradient_LGM, WTP_posterior_d_center_gradient_std_LGM, 'LGM', clim_LGM_WTP)
    column(d18O_CTP_avg, col_lons['CTP'], depth, vmin, vmax, levels, axes[1],
           CTP_posterior_d_center_gradient_LGM, CTP_posterior_d_center_gradient_std_LGM, 'LGM', clim_LGM_CTP)
    column(d18O_ETP_avg, col_lons['ETP'], depth, vmin, vmax, levels, axes[1],
           ETP_posterior_d_center_gradient_LGM, ETP_posterior_d_center_gradient_std_LGM, 'LGM', clim_LGM_ETP)

    # create arrays of size (1000,) to store the sum of d18O of the three regions
    d18O_WTP = np.zeros(1000)
    d18O_CTP = np.zeros(1000)
    d18O_ETP = np.zeros(1000)
    d18O_WTP_count = 0
    d18O_CTP_count = 0
    d18O_ETP_count = 0
    d_center_gradient_WTP = 0
    d_center_gradient_CTP = 0
    d_center_gradient_ETP = 0
    d_center_gradient_sum_sqr_WTP = 0
    d_center_gradient_sum_sqr_CTP = 0
    d_center_gradient_sum_sqr_ETP = 0

    # get Hol thermocline depth
    for itrace_file in itrace_file_Hol:
        # get core name
        core_name = itrace_file.removesuffix('_Hol.nc')

        # get core location
        site_lon, site_lat, site_depth = pseudo_d18Ocline.fetch_location(core_name, include_depth=True)

        # read itrace
        idata_d18Ocline_model_Hol = az.from_netcdf(os.path.join(itrace_path, itrace_file))
        # get idata parameters
        d_center_gradient = idata_d18Ocline_model_Hol.posterior['d_center_gradient'].median().values
        d18O_min = idata_d18Ocline_model_Hol.posterior['d18O_min'].median().values
        d18O_max = idata_d18Ocline_model_Hol.posterior['d18O_max'].median().values
        tilt = idata_d18Ocline_model_Hol.posterior['tilt'].median().values
        thickness = idata_d18Ocline_model_Hol.posterior['thickness'].median().values
        thermocline_std = idata_d18Ocline_model_Hol.posterior['d_center_gradient'].std().values

        # get predicted d18O profile
        depth, d18O = pseudo_d18Ocline.d18Ocline(d18O_min,
                                                 d18O_max,
                                                 1000,
                                                 d_center_gradient,
                                                 thickness,
                                                 tilt)

        # filter for WTP
        if site_lon > (160-25) and site_lon < (160+25) and site_lat > lat_S_lim and site_lat < lat_N_lim:
            d18O_WTP += d18O
            d18O_WTP_count += 1
            d_center_gradient_WTP += d_center_gradient
        elif site_lon > (-160-10) and site_lon < (-160+10) and site_lat > lat_S_lim and site_lat < lat_N_lim:
            d18O_CTP += d18O
            d18O_CTP_count += 1
            d_center_gradient_CTP += d_center_gradient
        elif site_lon > (-95-25) and site_lon < (-95+25) and site_lat > lat_S_lim and site_lat < lat_N_lim:
            d18O_ETP += d18O
            d18O_ETP_count += 1
            d_center_gradient_ETP += d_center_gradient
    # get d18O profile means
    d18O_WTP_avg = d18O_WTP / d18O_WTP_count
    d18O_CTP_avg = d18O_CTP / d18O_CTP_count
    d18O_ETP_avg = d18O_ETP / d18O_ETP_count

    # --- Interpolate Holocene climatology thermocline depths at column longitudes ---
    clim_Hol_WTP = float(np.interp(col_lons['WTP'], transect_df['lon'], transect_df['d_center_gradient']))
    clim_Hol_CTP = float(np.interp(col_lons['CTP'], transect_df['lon'], transect_df['d_center_gradient']))
    clim_Hol_ETP = float(np.interp(col_lons['ETP'], transect_df['lon'], transect_df['d_center_gradient']))

    # pymc
    idata_WTP_Hol = pseudo_d18Ocline.d18Ocline_pymc_simple(depth, d18O_WTP_avg)
    WTP_posterior_d_center_gradient_Hol = idata_WTP_Hol.posterior['d_center_gradient'].median().values
    WTP_posterior_d_center_gradient_std_Hol = idata_WTP_Hol.posterior['d_center_gradient'].std().values

    idata_CTP_Hol = pseudo_d18Ocline.d18Ocline_pymc_simple(depth, d18O_CTP_avg)
    CTP_posterior_d_center_gradient_Hol = idata_CTP_Hol.posterior['d_center_gradient'].median().values
    CTP_posterior_d_center_gradient_std_Hol = idata_CTP_Hol.posterior['d_center_gradient'].std().values

    idata_ETP_Hol = pseudo_d18Ocline.d18Ocline_pymc_simple(depth, d18O_ETP_avg)
    ETP_posterior_d_center_gradient_Hol = idata_ETP_Hol.posterior['d_center_gradient'].median().values
    ETP_posterior_d_center_gradient_std_Hol = idata_ETP_Hol.posterior['d_center_gradient'].std().values

    # plot the three columns
    column(d18O_WTP_avg, col_lons['WTP'], depth, vmin, vmax, levels, axes[0],
           WTP_posterior_d_center_gradient_Hol, WTP_posterior_d_center_gradient_std_Hol, 'Hol', clim_Hol_WTP)
    column(d18O_CTP_avg, col_lons['CTP'], depth, vmin, vmax, levels, axes[0],
           CTP_posterior_d_center_gradient_Hol, CTP_posterior_d_center_gradient_std_Hol, 'Hol', clim_Hol_CTP)
    column(d18O_ETP_avg, col_lons['ETP'], depth, vmin, vmax, levels, axes[0],
           ETP_posterior_d_center_gradient_Hol, ETP_posterior_d_center_gradient_std_Hol, 'Hol', clim_Hol_ETP)

    print(f'ETP deviation {clim_Hol_ETP - ETP_posterior_d_center_gradient_Hol}')
    print(f'CTP deviation {clim_Hol_CTP - CTP_posterior_d_center_gradient_Hol}')
    print(f'WTP deviation {clim_Hol_WTP - WTP_posterior_d_center_gradient_Hol}')
    print('==================')
    print(f'ETP posterior LGM change {ETP_posterior_d_center_gradient_LGM - ETP_posterior_d_center_gradient_Hol}')
    print(f'CTP posterior LGM change {CTP_posterior_d_center_gradient_LGM - CTP_posterior_d_center_gradient_Hol}')
    print(f'WTP posterior LGM change {WTP_posterior_d_center_gradient_LGM - WTP_posterior_d_center_gradient_Hol}')
    print('==================')
    print(f'ETP posterior LGM change std {np.sqrt(ETP_posterior_d_center_gradient_std_LGM**2 + ETP_posterior_d_center_gradient_std_Hol**2)}')
    print(f'CTP posterior LGM change std {np.sqrt(CTP_posterior_d_center_gradient_std_LGM**2 + CTP_posterior_d_center_gradient_std_Hol**2)}')
    print(f'WTP posterior LGM change std {np.sqrt(WTP_posterior_d_center_gradient_std_LGM**2 + WTP_posterior_d_center_gradient_std_Hol**2)}')

    # =========================================================================
    # THIRD ROW: MULTI-MODEL-MEAN dT/dz ANOMALY PANELS (c, d)
    # =========================================================================

    dT_dz_pi   = {}   # per-model dT/dz on the common grid
    dT_dz_lgm  = {}
    thermo_pi  = {}   # per-model thermocline on target_lon
    thermo_lgm = {}

    for model, info in models.items():
        print(f"Processing {model}...")
        try:
            # ── dT/dz from NetCDF ──────────────────────────────────────────
            ds_pi  = xr.open_dataset(info['pi'])
            ds_lgm = xr.open_dataset(info['lgm'])

            dT_dz_pi[model]  = extract_and_grid_transect(ds_pi,  info)
            dT_dz_lgm[model] = extract_and_grid_transect(ds_lgm, info)

            ds_pi.close()
            ds_lgm.close()

            # ── d_center_gradient thermocline from xlsx ────────────────────
            xf = xlsx_files[model]
            thermo_pi[model]  = load_thermocline_from_xlsx(xf['pi'])
            thermo_lgm[model] = load_thermocline_from_xlsx(xf['lgm'], xf.get('lgm2'))

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    mmm_axes    = [ax_l, ax_r]
    panel_letts = ['c', 'd']
    c = None
    for ax, (title, group), lett in zip(mmm_axes, panel_groups, panel_letts):
        # Keep only models that processed successfully
        members = [m for m in group if m in dT_dz_pi]
        print(f"Panel '{title}' from: {members}")

        # ── Group-mean dT/dz anomaly (sign convention: PI − LGM) ───────────
        group_anoms = [-(dT_dz_lgm[m] - dT_dz_pi[m]) for m in members]
        anomaly = np.nanmean(group_anoms, axis=0)

        # ── Group-mean thermoclines ────────────────────────────────────────
        mmm_thermo_pi  = np.nanmean([thermo_pi[m]  for m in members], axis=0)
        mmm_thermo_lgm = np.nanmean([thermo_lgm[m] for m in members], axis=0)

        # ── Regional-mean LGM − piControl thermocline anomaly ──────────────
        # WTP (<190°E), CTP (190–235°E), ETP (>235°E), within the plotted
        # transect domain (150–281°E). Positive = deeper thermocline at LGM.
        thermo_anom = mmm_thermo_lgm - mmm_thermo_pi
        domain = (target_lon >= 150) & (target_lon <= 281)
        regions = {
            'WTP': domain & (target_lon < 190),
            'CTP': domain & (target_lon >= 190) & (target_lon < 235),
            'ETP': domain & (target_lon >= 235),
        }
        print(f"--- {title}: MMM LGM - piControl thermocline anomaly (m) ---")
        for region, mask in regions.items():
            print(f'  {region}: {np.nanmean(thermo_anom[mask]):.2f}')

        # ── Background anomaly ─────────────────────────────────────────────
        c = ax.contourf(
            target_lon, target_depth, anomaly,
            levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r', extend='both'
        )

        # ── Thermocline lines from d_center_gradient (xlsx) ────────────────
        ax.plot(target_lon, mmm_thermo_pi,  'k-.', linewidth=1.5, label='piControl')
        ax.plot(target_lon, mmm_thermo_lgm, 'b-.', linewidth=1.5, label='LGM')

        ax.set_ylim(225, 0)          # 0 m at top, 300 m at bottom
        ax.set_xlim(150, 281)
        ax.set_xlabel('Longitude (°E)')
        ax.set_title(title, loc='right', fontsize=6)
        ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
        ax.text(0.02, 0.9, lett, fontweight='bold', transform=ax.transAxes)

    # ax_l.set_ylabel('Depth (m)')
    ax_r.set_yticklabels([])
    ax_l.legend(loc='lower right')

    # dT/dz colorbar for the two MMM panels
    cbar2 = fig.colorbar(c, cax=cax_bot)
    cbar2.set_label('dT/dz Anomaly (°C/m)')

    fig.set_size_inches(7, 5)

    # Re-size the arrowheads now that the figure has its final size and
    # constrained_layout has settled, so the head/stem ratio is right on output.
    fig.canvas.draw()
    for resize in arrow_resizers:
        resize()

    plt.savefig('Transect_core_regional_mean_overlay_MMM.png', dpi=700)
    # plt.savefig('Transect_core_regional_mean_overlay_MMM.pdf')
