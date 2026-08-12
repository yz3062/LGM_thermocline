#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-06-29

@author: zhou

Climatology-template warp for fitting multispecies foraminiferal d18O profiles.

Motivation
----------
pseudo_d18Ocline.d18Ocline_pymc() fits each core's d18O with a single
erf + quadratic curve (pseudo_d18Ocline.d18Ocline). That shape is a single
sigmoid and cannot reproduce the double-thermocline structure of the eastern
tropical Pacific (sharp 0-150 m gradient, a flattening "shadow zone" near
150-300 m, then a deeper gradient).

This module instead uses the local predicted-foram-d18O climatology profile
C(z) from Predicted_d18O.nc as a FIXED SHAPE TEMPLATE and fits a low-dimensional
warp of it to the foram observations:

    P(z) = a * C(z - z0) + b

    z0 : vertical depth shift (m), positive = climatology thermocline deepened
    a  : d18O amplitude (gradient contrast) scaling
    b  : d18O offset (absorbs glacial ice-volume / local mean shift)

The vertical stretch s is fixed to 1. This preserves the (double-)thermocline
shape inherent in C(z) while letting magnitude and depth positioning adapt to
the foram data. It works for both Holocene and LGM (the modern climatology
shape is assumed to hold at the LGM as a first-order assumption; b absorbs the
mean glacial shift) and for all tropical-Pacific sites.

This module does NOT edit pseudo_d18Ocline.py. It imports that module only to
reuse its read-only helpers (species_to_ACD, species_to_ACD_sigma,
find_nearest_valid_profile, ...). The ACD + benthic-anchor block is duplicated
here so the original module is untouched.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import pseudo_d18Ocline as pdc  # read-only helpers only


def _assemble_depth_d18O(d18O, d18O_sigma, species, Hol_or_LGM):
    '''
    Convert species to apparent calcification depth (ACD), then append the
    benthic anchor(s). Duplicated from pseudo_d18Ocline.d18Ocline_pymc so the
    original module is not modified.

    Parameters
    ----------
    d18O : pandas.Series
        Measured foraminifera d18O (permil).
    d18O_sigma : pandas.Series
        1-sigma of the measured foraminifera d18O (permil).
    species : pandas.DataFrame
        Single-column ('Species') frame, same length as d18O.
    Hol_or_LGM : str
        'Hol'/'Holocene' or 'LGM'. Determines the benthic d18O / depth anchors.

    Returns
    -------
    (depth_Series, depth_sigma_Series, d18O_Series, d18O_sigma_Series)
    '''
    # convert species to depth
    depth_Series = pdc.species_to_ACD(species, d18O, Hol_or_LGM)
    depth_sigma_Series = pdc.species_to_ACD_sigma(species)

    # add benthic
    if Hol_or_LGM == 'Hol' or Hol_or_LGM == 'Holocene':
        # depth
        Hol_benthic_depth = pd.Series([373, 617])  # from Bova et al. 2015 : KNR195-5 GGC43
        depth_Series = pd.concat([depth_Series, Hol_benthic_depth])
        # depth sigma
        Hol_benthic_depth_sigma = pd.Series([1, 1])  # benthic depth should have very little uncertainty
        depth_sigma_Series = pd.concat([depth_sigma_Series, Hol_benthic_depth_sigma])
        # d18O
        Hol_benthic_d18O = pd.Series([1.34, 1.75])
        d18O_Series = pd.concat([d18O, Hol_benthic_d18O])
        # d18O sigma
        Hol_benthic_d18O_sigma = pd.Series([0.162, 0.154])  # sqrt(0.15^2+0.053^2+0.03^2) and sqrt(0.15^2+0.020^2+0.03^2)
        d18O_sigma_Series = pd.concat([d18O_sigma, Hol_benthic_d18O_sigma])
    elif Hol_or_LGM == 'LGM':
        # depth
        LGM_benthic_depth = pd.Series([505.1])  # from Bova et al. 2015 : KNR195-5 GGC43
        depth_Series = pd.concat([depth_Series, LGM_benthic_depth])
        # depth sigma
        LGM_benthic_depth_sigma = pd.Series([5])  # LGM depth is a bit more uncertain due to sea level
        depth_sigma_Series = pd.concat([depth_sigma_Series, LGM_benthic_depth_sigma])
        # d18O
        LGM_benthic_d18O = pd.Series([2.79])
        d18O_Series = pd.concat([d18O, LGM_benthic_d18O])
        # d18O sigma
        LGM_benthic_d18O_sigma = pd.Series([0.155])  # sqrt(0.15^2+0.024^2+0.03^2)
        d18O_sigma_Series = pd.concat([d18O_sigma, LGM_benthic_d18O_sigma])
    else:
        raise ValueError('Period must be either Hol or LGM')

    return (depth_Series, depth_sigma_Series, d18O_Series, d18O_sigma_Series)


def _build_template(lon, lat, predicted_d18O_ds, radius=5, d_max=2000, dz=1.0):
    '''
    Build a dense, regular-grid version of the local climatology profile C(z)
    so it can be used as a fixed interpolation template.

    Parameters
    ----------
    lon, lat : float
        Site coordinates.
    predicted_d18O_ds : xarray.Dataset
        xr.load_dataset('Predicted_d18O.nc', decode_times=False).
    radius : float
        Search radius (deg) for the nearest valid profile.
    d_max : float
        Maximum depth (m) of the template grid.
    dz : float
        Grid spacing (m).

    Returns
    -------
    zf : np.ndarray
        Regular depth grid, 0 .. d_max (m).
    Cf : np.ndarray
        Climatology d18O interpolated onto zf (ends constant-extrapolated).
    z_tc_clim : float
        Depth (m) of maximum dC/dz of the template = the climatology's main
        (upper) thermocline depth.
    '''
    prof = pdc.find_nearest_valid_profile(predicted_d18O_ds, lon, lat,
                                          'd18O_foram', radius)
    prof = prof.dropna(subset=['d18O_foram']).sort_values('depth')
    z_clim = prof['depth'].values.astype(float)
    C_clim = prof['d18O_foram'].values.astype(float)

    zf = np.arange(0.0, d_max + dz, dz)
    # np.interp constant-extrapolates beyond the data range (returns end values)
    Cf = np.interp(zf, z_clim, C_clim)

    # max-gradient depth of the template (a/b do not move this argmax)
    z_tc_clim = float(zf[np.argmax(np.gradient(Cf, zf))])

    return zf, Cf, z_tc_clim


def d18Ocline_pymc_climatology(d18O, d18O_sigma, species, Hol_or_LGM,
                               lon, lat, predicted_d18O_ds,
                               radius=5, d_max=2000, dz=1.0):
    '''
    Fit the local climatology profile, warped as P(z) = a * C(z - z0) + b, to
    multispecies foram d18O observations. Drop-in alternative to
    pseudo_d18Ocline.d18Ocline_pymc for sites where the erf + quadratic shape
    (notably the eastern Pacific double thermocline) fails.

    Parameters
    ----------
    d18O : pandas.Series
        Measured foraminifera d18O (permil).
    d18O_sigma : pandas.Series
        1-sigma of the measured foraminifera d18O (permil).
    species : pandas.DataFrame
        Single-column ('Species') frame, same length as d18O.
    Hol_or_LGM : str
        'Hol'/'Holocene' or 'LGM'. Sets the benthic anchors.
    lon, lat : float
        Site coordinates (used to look up the local climatology template).
    predicted_d18O_ds : xarray.Dataset
        xr.load_dataset('Predicted_d18O.nc', decode_times=False).
    radius : float
        Search radius (deg) for the nearest valid climatology profile.
    d_max : float
        Maximum depth (m) of the template grid (>= deepest benthic anchor).
    dz : float
        Template grid spacing (m).

    Returns
    -------
    (idata_d18Ocline_model, depth_Series, depth_sigma_Series,
     d18O_Series, d18O_sigma_Series)
        Mirrors pseudo_d18Ocline.d18Ocline_pymc's return signature.
        idata_d18Ocline_model.posterior exposes z0, a, b and the Deterministic
        d_center_gradient (= z_tc_clim + z0), the thermocline-depth metric used
        downstream.
    '''
    # ACD + benthic anchoring (same as d18Ocline_pymc)
    (depth_Series, depth_sigma_Series,
     d18O_Series, d18O_sigma_Series) = _assemble_depth_d18O(
        d18O, d18O_sigma, species, Hol_or_LGM)

    # local climatology shape template (constant data into the graph)
    zf, Cf, z_tc_clim = _build_template(lon, lat, predicted_d18O_ds,
                                        radius=radius, d_max=d_max, dz=dz)

    depth_obs = depth_Series.values.astype(float)
    depth_obs_sigma = depth_sigma_Series.values.astype(float)
    d18O_obs = d18O_Series.values.astype(float)
    d18O_obs_sigma = d18O_sigma_Series.values.astype(float)

    z_lo = float(zf[0])
    z_hi = float(zf[-1])
    n_knot = len(zf)

    with pm.Model() as d18Ocline_model:
        # Priors for the warp
        z0 = pm.TruncatedNormal("z0", mu=0.0, sigma=50.0, lower=-z_tc_clim)  # depth shift (m); lower bound keeps d_center_gradient >= 0
        a = pm.TruncatedNormal("a", mu=1.0, sigma=0.3, lower=0.0)      # contrast scaling
        b = pm.Normal("b", mu=0.0, sigma=1.0)                          # d18O offset

        # Latent (uncertain) observation depths, as in d18Ocline_pymc
        depth_latent = pm.Normal("depth_latent", mu=depth_obs, sigma=depth_obs_sigma)

        # Piecewise-linear interpolation of the constant template Cf at the
        # warped depths q = depth_latent - z0. The template is constant, so the
        # gradient of mu w.r.t. q is exactly the local template slope
        # a * (Cf[i0+1] - Cf[i0]) / dz, flowing through `frac`; the discrete
        # index i0 (via floor) carries no gradient, which is correct.
        Cf_t = pt.as_tensor_variable(Cf)
        q = pt.clip(depth_latent - z0, z_lo, z_hi)
        qi = (q - z_lo) / dz
        i0 = pt.clip(pt.cast(pt.floor(qi), "int64"), 0, n_knot - 2)
        frac = qi - pt.cast(i0, "floatX")
        Cval = Cf_t[i0] * (1.0 - frac) + Cf_t[i0 + 1] * frac

        # Warped profile evaluated at the observation depths
        mu = a * Cval + b

        # Thermocline-depth metric for downstream compatibility
        pm.Deterministic("d_center_gradient", z_tc_clim + z0)

        # Likelihood
        pm.Normal("Y_observs", mu=mu, sigma=d18O_obs_sigma, observed=d18O_obs)

    with d18Ocline_model:
        idata_d18Ocline_model = pm.sample(target_accept=0.99, progressbar=False)

    return (idata_d18Ocline_model, depth_Series, depth_sigma_Series,
            d18O_Series, d18O_sigma_Series)


def warp_climatology_profile(idata, lon, lat, predicted_d18O_ds,
                             radius=5, d_max=2000, dz=1.0):
    '''
    Reconstruct the fitted warped profile P(z) = a * C(z - z0) + b on a depth
    grid using posterior median parameters, for plotting. Mirrors the return of
    pseudo_d18Ocline.d18Ocline (depth, d18O), so it can be dropped into the
    existing plotting blocks in place of d18Ocline().

    Parameters
    ----------
    idata : arviz.InferenceData
        Return of d18Ocline_pymc_climatology.
    lon, lat : float
        Site coordinates (same as used in the fit).
    predicted_d18O_ds : xarray.Dataset
        Predicted_d18O.nc dataset.
    radius, d_max, dz : see d18Ocline_pymc_climatology (must match the fit).

    Returns
    -------
    (depth, d18O) : np.ndarray, np.ndarray
        depth (m, increasing) and the reconstructed warped d18O profile.
    '''
    zf, Cf, _ = _build_template(lon, lat, predicted_d18O_ds,
                                radius=radius, d_max=d_max, dz=dz)

    z0 = float(idata.posterior['z0'].median().values)
    a = float(idata.posterior['a'].median().values)
    b = float(idata.posterior['b'].median().values)

    # P(z) = a * C(z - z0) + b, evaluated on the template grid
    q = np.clip(zf - z0, zf[0], zf[-1])
    Cval = np.interp(q, zf, Cf)
    d18O = a * Cval + b

    return zf, d18O
