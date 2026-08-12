#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-07-09

@author: zhou

Fits the erf + quadratic d18Ocline model (pseudo_d18Ocline.d18Ocline_pymc_simple)
to the reconstructed warped-climatology profile of each core in
Core_d18O_pymc_inference_climatology (built with
pseudo_d18Ocline_climatology.warp_climatology_profile). This gives an
erf-derived thermocline depth (d_center_gradient) for a profile that already
carries the climatology-template shape, for comparison against the warp
method's own z0-based metric.

Adapted from pseudo_d18Ocline_pymc_calculation_parallel_Hol.py, but iterates
over the existing climatology-warp inference files instead of re-deriving a
core list from the raw d18O spreadsheets.
"""

import os
import arviz as az
import matplotlib.pyplot as plt
import pseudo_d18Ocline
import pseudo_d18Ocline_climatology
import xarray as xr
import concurrent.futures

SRC_DIR = 'Core_d18O_pymc_inference_climatology'
INFERENCE_OUT_DIR = 'Core_d18O_pymc_inference_on_warped_climatology'
FIGURE_OUT_DIR = 'Core_d18O_pymc_figure_erf_on_warped_climatology'

os.makedirs(INFERENCE_OUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUT_DIR, exist_ok=True)

# predicted foram d18O climatology (module-level so it survives the spawn
# re-import used by ProcessPoolExecutor)
predicted_d18O_ds = xr.load_dataset('Predicted_d18O.nc', decode_times=False)

itrace_files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith('.nc'))


def core_name_and_period_from_file(filename):
    for suffix, period in (('_Hol.nc', 'Hol'), ('_LGM.nc', 'LGM')):
        if filename.endswith(suffix):
            return filename[: -len(suffix)], period
    raise ValueError(f'Unrecognized file name pattern: {filename}')


def d18O_cline_calc(i):
    filename = itrace_files[i]
    core_name, Hol_or_LGM = core_name_and_period_from_file(filename)
    
    site_lon, site_lat = pseudo_d18Ocline.fetch_location(core_name, False)[:2]

    print('Working on: ' + str(core_name) + '_' + Hol_or_LGM)

    idata_climatology = az.from_netcdf(os.path.join(SRC_DIR, filename))
    depth, d18O_profile = pseudo_d18Ocline_climatology.warp_climatology_profile(
        idata_climatology, site_lon, site_lat, predicted_d18O_ds)

    d_max = float(depth.max())
    idata_erf_model = pseudo_d18Ocline.d18Ocline_pymc_simple(
        depth, d18O_profile, d_max=d_max)
    idata_erf_model.to_netcdf(os.path.join(INFERENCE_OUT_DIR, core_name + '_' + Hol_or_LGM + '.nc'))

    az.plot_trace(idata_erf_model, combined=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUT_DIR, core_name + '_' + Hol_or_LGM + '_itrace.png'), dpi=70)
    plt.close('all')

    #%%
    plt.figure()

    # plot the warped-climatology profile being fit
    plt.plot(d18O_profile, depth, label='Warped climatology', c='C0')

    # plot posterior erf fit
    depth_fit, d18O_fit = pseudo_d18Ocline.d18Ocline(
        idata_erf_model.posterior['d18O_min'].median().values,
        idata_erf_model.posterior['d18O_max'].median().values,
        d_max,
        idata_erf_model.posterior['d_center_gradient'].median().values,
        idata_erf_model.posterior['thickness'].median().values,
        idata_erf_model.posterior['tilt'].median().values)
    plt.plot(d18O_fit, depth_fit, label='erf fit', c='C1')

    plt.xlim(-3.5, 3.5)
    plt.ylim(0, 1000)
    plt.xlabel('d18O (permil)')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.ylabel('Depth (m)')
    plt.title('Core name: ' + str(core_name) + '_' + Hol_or_LGM)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.gcf().set_size_inches(4, 6)
    plt.savefig(os.path.join(FIGURE_OUT_DIR, core_name + '_' + Hol_or_LGM + '_profile.png'), dpi=70)
    plt.close('all')


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as pool:
        list(pool.map(d18O_cline_calc, range(len(itrace_files))))
