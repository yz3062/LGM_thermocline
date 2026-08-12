#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-06-29

@author: zhou

Climatology-template version of pseudo_d18Ocline_pymc_calculation_parallel_LGM.py.

Instead of fitting the erf + quadratic d18Ocline function, this fits a warped
local climatology profile (P(z) = a * C(z - z0) + b) from Predicted_d18O.nc to
each core's multispecies foram d18O. This captures the eastern Pacific double
thermocline that the parametric function cannot. The modern climatology shape is
assumed to hold at the LGM; the fitted d18O offset b absorbs the mean glacial
(ice-volume / local) shift. Results are saved as separate netCDF inference files
(new output dirs) so the erf-based outputs are untouched.
"""

import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pseudo_d18Ocline
import pseudo_d18Ocline_climatology
import xarray as xr
import concurrent.futures
import itertools

# predicted foram d18O climatology (module-level so it survives the spawn
# re-import used by ProcessPoolExecutor)
predicted_d18O_ds = xr.load_dataset('Predicted_d18O.nc', decode_times=False)

# thermocline fit
Hol_or_LGM = 'LGM'
if Hol_or_LGM == 'Hol':
    d18O_pd = pseudo_d18Ocline.load_Hol_d18O('average')
elif Hol_or_LGM == 'LGM':
    d18O_pd = pseudo_d18Ocline.load_LGM_d18O('average')
d18O_pd_grouped = d18O_pd.groupby(['CoreName']).size().sort_values(ascending=False).to_frame() # to use iterrows()
# filter for more than 2 species
d18O_pd_grouped = d18O_pd_grouped[d18O_pd_grouped>=2].dropna()

def d18O_cline_calc(i):
    core_name = d18O_pd_grouped.index[i]
    d18O = d18O_pd[d18O_pd['CoreName']==core_name]['d18O']
    d18O_sigma = d18O_pd[d18O_pd['CoreName']==core_name]['d18O_1sigma']
    species = d18O_pd[d18O_pd['CoreName']==core_name]['Species'].to_frame() # to use merge
    
    # filter only those cores with two or more subsurface species
    mask = species['Species'].str.contains('G. tumida').astype(int) +\
            species['Species'].str.contains('N. dutertrei').astype(int) +\
            species['Species'].str.contains('P. obliquiloculata').astype(int)

    # Filter for rows where at least 2 conditions were met
    if mask.sum() < 2:
        return f"Skipped {core_name}: insufficient species"

    # site location for the climatology template lookup
    lon, lat = pseudo_d18Ocline.fetch_location(core_name, False)[:2]

    # printout
    print('Working on: ' + str(core_name))
    print('d18O: ' + d18O.to_string())
    print('d18O sigma: ' + d18O_sigma.to_string())
    print('Species: ' + species.to_string())

    idata_d18Ocline_model, depth_Series, depth_sigma_Series, d18O_Series, d18O_sigma_Series = pseudo_d18Ocline_climatology.d18Ocline_pymc_climatology(d18O, d18O_sigma, species, Hol_or_LGM, lon, lat, predicted_d18O_ds)
    idata_d18Ocline_model.to_netcdf('./Core_d18O_pymc_inference_climatology/'+str(core_name)+'_'+Hol_or_LGM+'.nc')

    az.plot_trace(idata_d18Ocline_model, combined=True)
    plt.tight_layout()
    plt.savefig('./Core_d18O_pymc_figure_climatology/'+str(core_name)+'_'+Hol_or_LGM+'_itrace.png',dpi=70)

    #%%
    plt.figure()

    # plot observations
    plt.errorbar(d18O_Series,
                 depth_Series,
                 xerr=d18O_sigma_Series,
                 yerr=depth_sigma_Series,
                 label='Observation',
                 markerfacecolor='w',
                 markeredgecolor='C0',
                 ecolor='C0',
                 fmt='o', linewidth=1, capsize=3, markersize=5)
    plt.xlim(-3.5, 3.5)
    plt.ylim(0, 1000)
    plt.xlabel('d18O (permil)')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.ylabel('Depth (m)')
    plt.title('Core name: ' + str(core_name) + '_' + Hol_or_LGM)
    plt.gcf().set_size_inches(4,6)

    # plot posterior warped-climatology profile
    depth, d18O_profile = pseudo_d18Ocline_climatology.warp_climatology_profile(idata_d18Ocline_model, lon, lat, predicted_d18O_ds)
    plt.plot(d18O_profile, depth, label='Posterior thermocline', c='C1')

    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig('./Core_d18O_pymc_figure_climatology/'+str(core_name)+'_'+Hol_or_LGM+'_profile.png',dpi=70)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as pool:
        df_result_iter = pool.map(d18O_cline_calc, range(len(d18O_pd_grouped)))
