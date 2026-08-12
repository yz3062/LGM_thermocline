#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 18:07:23 2025

@author: zhou

This is a collection of functions
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import prior_from_idata
import xarray as xr
# import seaborn as sns
from pathlib import Path

# 1. Get the directory where THIS script (my_data_tools.py) is located
THIS_DIR = Path(__file__).resolve().parent

# sns.set(font='Arial',palette='husl',style='whitegrid',context='paper')#, font_scale=2)

def load_Hol_d18O(average_or_individual='average', keep_missing_depth=False):
    df = pd.read_excel(THIS_DIR / './Data/Lakhani_2024/Hol_d18O_literature_this_study.xlsx')
    df.rename(columns={'d18O_sigma': 'd18O_1sigma'}, inplace=True)
    df = df.dropna(subset=['d18O'])
    df['d18O_1sigma'] = df['d18O_1sigma'].fillna(0.05)
    if not keep_missing_depth:
        df = df.dropna(subset=['Depth (m)'])
        df['Depth (m)'] = pd.to_numeric(df['Depth (m)'])
    if average_or_individual == 'average':
        # some d18O values are the averages of multiple studies. Weigh them more
        df['d18O_weighted'] = df['d18O'] * df['Weight']
        d18O_weighted_mean = df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['d18O_weighted']/df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['Weight']
        # d18O sigma too
        df['d18O_1sigma_weighted'] = df['d18O_1sigma'] * df['Weight']
        d18O_1sigma_weighted_mean = df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['d18O_1sigma_weighted']/df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['Weight']
        # set up a new df as output since it's shorter in length
        # This is so all meta data is passed on
        # For string columns, include the first row
        df_output = df.groupby(['CoreName', 'Species']).agg({
            **{col: 'mean' for col in df.select_dtypes(include='number').columns},
            **{col: 'first' for col in df.select_dtypes(include='object').columns 
               if col not in ['CoreName', 'Species']}
        })
        df_output['d18O'] = d18O_weighted_mean
        df_output['d18O_1sigma'] = d18O_1sigma_weighted_mean
    return df_output.reset_index()

def load_LGM_d18O(average_or_individual='average'):
    df = pd.read_excel(THIS_DIR / './Data/Lakhani_chapter3/LGM_database_this_study.xlsx')
    df.dropna(subset=['d18O'], inplace=True)
    df['d18O_1sigma'] = df['d18O_1sigma'].fillna(0.05)
    if average_or_individual == 'average':
        # some d18O values are the averages of multiple studies. Weigh them more
        df['d18O_weighted'] = df['d18O'] * df['Weight']
        d18O_weighted_mean = df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['d18O_weighted']/df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['Weight']
        # d18O sigma too
        df['d18O_1sigma_weighted'] = df['d18O_1sigma'] * df['Weight']
        d18O_1sigma_weighted_mean = df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['d18O_1sigma_weighted']/df.groupby(['CoreName', 'Species']).sum(numeric_only=True)['Weight']
        # set up a new df as output since it's shorter in length
        # This is so all meta data is passed on
        # For string columns, include the first row
        df_output = df.groupby(['CoreName', 'Species']).agg({
            **{col: 'mean' for col in df.select_dtypes(include='number').columns},
            **{col: 'first' for col in df.select_dtypes(include='object').columns 
               if col not in ['CoreName', 'Species']}
        })
        df_output['d18O'] = d18O_weighted_mean
        df_output['d18O_1sigma'] = d18O_1sigma_weighted_mean
    return df_output.reset_index()

def fetch_location(CoreName, include_depth):
    if include_depth:
        d18O_pd_Hol = load_Hol_d18O('average', keep_missing_depth=False)
        core_df = d18O_pd_Hol[d18O_pd_Hol['CoreName'] == CoreName]
        if core_df.empty:
            d18O_pd_LGM = load_LGM_d18O('average')
            core_df = d18O_pd_LGM[d18O_pd_LGM['CoreName'] == CoreName]
            if core_df.empty:
                raise ValueError('Core ' + CoreName + ' not found in either LGM or Hol spreadsheet')
        return(core_df['Longitude'].values.mean(), core_df['Latitude'].values.mean(), core_df['Depth (m)'].values.mean())
    else:
        d18O_pd_Hol = load_Hol_d18O('average', keep_missing_depth=True)
        core_df = d18O_pd_Hol[d18O_pd_Hol['CoreName'] == CoreName]
        if core_df.empty:
            d18O_pd_LGM = load_LGM_d18O('average')
            core_df = d18O_pd_LGM[d18O_pd_LGM['CoreName'] == CoreName]
            if core_df.empty:
                raise ValueError('Core ' + CoreName + ' not found in either LGM or Hol spreadsheet')
        return(core_df['Longitude'].values.mean(), core_df['Latitude'].values.mean())
        
def fetch_d18O(CoreName, Hol_or_LGM):
    '''
    

    Parameters
    ----------
    CoreName : string
    Hol_or_LGM : string
        Hol or LGM profile. Determines the benthic d18O depth due to sea
        level change.

    Returns
    -------
    depth_Series, depth_sigma_Series,
            d18O_Series, d18O_sigma_Series

    '''
    
    if Hol_or_LGM == 'Hol' or Hol_or_LGM == 'Holocene':
        d18O_pd_Hol = load_Hol_d18O('average')
        core_df = d18O_pd_Hol[d18O_pd_Hol['CoreName'] == CoreName]
    elif Hol_or_LGM == 'LGM':
        d18O_pd_LGM = load_LGM_d18O()
        core_df = d18O_pd_LGM[d18O_pd_LGM['CoreName'] == CoreName]
    else:
        raise ValueError('Period must be either Hol or LGM')
    
    # convert species to depth
    d18O = core_df['d18O']
    d18O_sigma = core_df['d18O_1sigma']
    depth_Series = species_to_ACD(core_df['Species'].to_frame(), d18O, Hol_or_LGM)
    depth_sigma_Series = species_to_ACD_sigma(core_df['Species'].to_frame())
    
    # add benthic
    if Hol_or_LGM == 'Hol' or Hol_or_LGM == 'Holocene':
        # depth
        Hol_benthic_depth = pd.Series([373, 617]) # from Bova et al. 2015 : KNR195-5 GGC43
        depth_Series = pd.concat([depth_Series, Hol_benthic_depth])
        # depth sigma
        Hol_benthic_depth_sigma = pd.Series([1, 1]) # benthic depth should have very little uncertainty
        depth_sigma_Series = pd.concat([depth_sigma_Series, Hol_benthic_depth_sigma])
        # d18O
        Hol_benthic_d18O = pd.Series([1.34, 1.75])
        d18O_Series = pd.concat([d18O, Hol_benthic_d18O])
        # d18O sigma
        Hol_benthic_d18O_sigma = pd.Series([0.162, 0.154]) # sqrt(0.15^2+0.053^2+0.03^2) and sqrt(0.15^2+0.020^2+0.03^2) # Combine instrumental (Bova 2015), population (LeGrande and Schmidt 2006), and foram-seawater offset (Marchitto 2014) uncertainty
        d18O_sigma_Series = pd.concat([d18O_sigma, Hol_benthic_d18O_sigma])
    elif Hol_or_LGM == 'LGM':
        # depth
        LGM_benthic_depth = pd.Series([505.1]) # from Bova et al. 2015 : KNR195-5 GGC43
        depth_Series = pd.concat([depth_Series, LGM_benthic_depth])
        # depth sigma
        LGM_benthic_depth_sigma = pd.Series([5]) # LGM depth is a bit more uncertain due to sea level
        depth_sigma_Series = pd.concat([depth_sigma_Series, LGM_benthic_depth_sigma])
        # d18O
        LGM_benthic_d18O = pd.Series([2.79])
        d18O_Series = pd.concat([d18O, LGM_benthic_d18O])
        # d18O sigma
        LGM_benthic_d18O_sigma = pd.Series([0.155]) # sqrt(0.15^2+0.024^2+0.03^2) # Combine instrumental (Bova 2015), population (LeGrande and Schmidt 2006), and foram-seawater offset (Marchitto 2014) uncertainty
        d18O_sigma_Series = pd.concat([d18O_sigma, LGM_benthic_d18O_sigma])
    else:
        raise ValueError('Period must be either Hol or LGM')
    return (depth_Series, depth_sigma_Series,
            d18O_Series, d18O_sigma_Series)

def fetch_species(CoreName, Hol_or_LGM):
    '''
    

    Parameters
    ----------
    CoreName : string
    Hol_or_LGM : string
        Hol or LGM profile. Determines the benthic d18O depth due to sea
        level change.

    Returns
    -------
    species_Series

    '''
    
    if Hol_or_LGM == 'Hol' or Hol_or_LGM == 'Holocene':
        d18O_pd_Hol = load_Hol_d18O('average')
        core_df = d18O_pd_Hol[d18O_pd_Hol['CoreName'] == CoreName]
    elif Hol_or_LGM == 'LGM':
        d18O_pd_LGM = load_LGM_d18O()
        core_df = d18O_pd_LGM[d18O_pd_LGM['CoreName'] == CoreName]
    else:
        raise ValueError('Period must be either Hol or LGM')
    
    # convert species to depth
    species_Series = core_df['Species']
    size_fraction_Series = core_df['Size fraction (mm)']
    
    return (species_Series, size_fraction_Series)

def d18Ocline(d18O_min, d18O_max, d_max, d_center_gradient, thickness, tilt, N_points=1000):
    '''
    %% d18Ocline function:
    % Generates a d18O gradient that mimics a d18Ocline as observed by planktic and benthic foraminifera.
    %
    % INPUT : 
    %           d18O_min : minimum d18O
    %           d18O_max : maximum d18O of the gradient
    %           d_max : maximum depth of the output ( +meters )
    %           d_center_gradient : center of transition region ( +meters )
    %           thickness : Thickness of the transition region, this is based
    %                       on an estimate of the standard deviation.
    %           tilt: tilt above and below the thermocline. Have to be >=1          
    %
    %           optional-->
    %           N_points : Number of points of output desired, default=1000;
    %
    % OUTPUT :
    %           d18O : d18O (x-axis)
    %           depth : depth, negative values, from 0 to -DEPTH;
    %
    % EXAMPLE :
    %           To generate a d18Ocline between -2 and 2 permil, with
    %           the center of the transition at -200m, with approximately a
    %           transition width of 100 meters, a maximum depth of 1000
    %           meters, and using 3000 points, the command would be:
    %
    %           [temp, depth] = thermocline(-2,2,1000,200,100,3000);
    % By: Yuxin Zhou, Georgia Tech
    % Inspired by: Jared Brzenski, San Diego State University
    %
    '''
    
    
    # Settign up the variables for the erf
    depth = np.linspace(0,1,N_points);
    mu =  d_center_gradient/d_max ;
    sigma2 = thickness / d_max;    # Scale the thickness
    sigma2 = sigma2 / 2.3548 / 2 ; # FWHM estimate of SD
    denom = sigma2 * np.sqrt(2);      # sigma * sqrt(2)
    
    # temperature values calculated here
    d18O = d18O_min + (d18O_max-d18O_min-tilt)*0.5*(1 + special.erf( (depth - mu)/denom)) - tilt*(1-depth)**2 + tilt;
    
    # rescale depth from 0->1 to 0->-depth
    # depth = depth - 1;
    depth = depth * d_max;
    
    return np.array([depth, d18O])

def get_thermocline_depth_thickness_relation():
    '''
    

    Returns
    -------
    idata_fit_model : arviz.InferenceData
        Inference result.

    '''
    # read data
    df = pd.read_excel(THIS_DIR / 'Thermocline_depth_thickness.xlsx')
    
    # this is the predictor variable
    thermocline_depth = df['d_center_gradient']
    
    with pm.Model() as fit_model:
        alpha = pm.Normal("alpha", mu=6e-5, sigma=1e-5)
        beta = pm.Normal("beta", mu=0.0177, sigma=1e-2)
        gamma = pm.Normal("gamma", mu=2.3456, sigma=1)
        delta = pm.Normal("delta", mu=14.615, sigma=1)
        
        # polynomial equation
        mu = alpha * thermocline_depth**3 - beta * thermocline_depth**2 + gamma * thermocline_depth + delta
        
        # obs sigma
        sigma = pm.HalfNormal("sigma", sigma=1)
        
        # Likelihood (sampling distribution) of observations
        thickness_obs = pm.Normal("thickness_obs", mu=mu, sigma=sigma, observed=df['thickness'])
        
        # sample
    with fit_model:
        idata_fit_model = pm.sample()#target_accept=0.996)
    return idata_fit_model
        
def d18Ocline_pymc(idata_fit_model, d18O, d18O_sigma, species, Hol_or_LGM, d_max=1000):
    '''
    Unlike d18Ocline_pymc_simple(),
    this function is for fitting core d18O measurements

    Parameters
    ----------
    idata_fit_model : arviz.data.inference_data.InferenceData
        Return from get_thermocline_depth_thickness_relation().
    d18O : Array
        Measured foraminifera d18O (permil).
    d18O_sigma : Double
        Standard deviation of the measured foraminifera d18O (permil).
    species : Array
        The species of foraminifera of each d18O. Must have the same shape as
        d18O. This is used to generate the apparent carcification depth (ACD)
        of each d18O data point.
    d_max : int
        Max depth (m) of the pseudo d18O profile. Somewhere between 600 and
        1000 m is appropriate.
    Hol_or_LGM : Boolean
        Hol or LGM profile. Determines the benthic d18O depth due to sea
        level change.

    Returns
    -------
    idata_d18Ocline_model : arviz.InferenceData
        Inference result.

    '''
    
    # convert species to depth
    depth_Series = species_to_ACD(species, d18O, Hol_or_LGM)
    depth_sigma_Series = species_to_ACD_sigma(species)
    
    # add benthic
    if Hol_or_LGM == 'Hol' or Hol_or_LGM == 'Holocene':
        # depth
        Hol_benthic_depth = pd.Series([373, 617]) # from Bova et al. 2015 : KNR195-5 GGC43
        depth_Series = pd.concat([depth_Series, Hol_benthic_depth])
        # depth sigma
        Hol_benthic_depth_sigma = pd.Series([1, 1]) # benthic depth should have very little uncertainty
        depth_sigma_Series = pd.concat([depth_sigma_Series, Hol_benthic_depth_sigma])
        # d18O
        Hol_benthic_d18O = pd.Series([1.34, 1.75])
        d18O_Series = pd.concat([d18O, Hol_benthic_d18O])
        # d18O sigma
        Hol_benthic_d18O_sigma = pd.Series([0.162, 0.154]) # sqrt(0.15^2+0.053^2+0.03^2) and sqrt(0.15^2+0.020^2+0.03^2) # Combine instrumental (Bova 2015), population (LeGrande and Schmidt 2006), and foram-seawater offset (Marchitto 2014) uncertainty
        d18O_sigma_Series = pd.concat([d18O_sigma, Hol_benthic_d18O_sigma])
    elif Hol_or_LGM == 'LGM':
        # depth
        LGM_benthic_depth = pd.Series([505.1]) # from Bova et al. 2015 : KNR195-5 GGC43
        depth_Series = pd.concat([depth_Series, LGM_benthic_depth])
        # depth sigma
        LGM_benthic_depth_sigma = pd.Series([5]) # LGM depth is a bit more uncertain due to sea level
        depth_sigma_Series = pd.concat([depth_sigma_Series, LGM_benthic_depth_sigma])
        # d18O
        LGM_benthic_d18O = pd.Series([2.79])
        d18O_Series = pd.concat([d18O, LGM_benthic_d18O])
        # d18O sigma
        LGM_benthic_d18O_sigma = pd.Series([0.155]) # sqrt(0.15^2+0.024^2+0.03^2) # Combine instrumental (Bova 2015), population (LeGrande and Schmidt 2006), and foram-seawater offset (Marchitto 2014) uncertainty
        d18O_sigma_Series = pd.concat([d18O_sigma, LGM_benthic_d18O_sigma])
    else:
        raise ValueError('Period must be either Hol or LGM')
    
    # # get the length of data
    # d18O_len = len(d18O_Series)
    
    # private variable. Scaled depth
    __depth = depth_Series / d_max
    __depth_sigma = depth_sigma_Series / d_max
    
    # set up pymc model
    with pm.Model() as d18Ocline_model:
        # priors from the fit_model
        priors_fit_model = prior_from_idata.prior_from_idata(idata_fit_model, var_names=['alpha', 'beta', 'gamma', 'delta'])
        
        # Priors for unknown model parameters
        d_center_gradient = pm.Normal("d_center_gradient", mu=150, sigma=50)
        # thickness = priors_fit_model['alpha'] * d_center_gradient**3 - priors_fit_model['beta'] * d_center_gradient**2 + priors_fit_model['gamma'] * d_center_gradient + priors_fit_model['delta']
        thickness = pm.Normal("thickness", mu=150, sigma=50)
        # tilt is lognormal to prevent negative values
        tilt = pm.TruncatedNormal("tilt", mu=1.3, sigma=0.2, lower=0, upper=3)
        d18O_min = pm.Normal("d18O_min", mu=np.min(d18O_Series), sigma=0.05)
        d18O_max = pm.Normal("d18O_max", mu=np.max(d18O_Series), sigma=0.2)
        
        # Priors for depth
        depth_latent = pm.Normal('depth_latent', mu=__depth, sigma=__depth_sigma)# , shape=d18O_len)
        
        # known parameters
        __mu = d_center_gradient / d_max # scaled d center gradient
        __denom = thickness / 2.3548 / 2 / d_max * pm.math.sqrt(2)      # sigma * sqrt(2)
        # not using as sigma to y_obs because not needing to estimate sigma
        # sigma = pm.HalfNormal("sigma", sigma=d18O_sigma_Series)#, shape=d18O_len)
        
        # Expected value of outcome
        mu = d18O_min + (d18O_max-d18O_min-tilt)*0.5*(1 + pm.math.erf( (depth_latent - __mu)/__denom)) - tilt*(1-depth_latent)**2 + tilt
    
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_observs", mu=mu, sigma=d18O_sigma_Series, observed=d18O_Series)
        
    with d18Ocline_model:
        idata_d18Ocline_model = pm.sample(target_accept=0.996, progressbar=False)#target_accept=0.996) # cores=1: a bug prevents multiprocessing
        
    return (idata_d18Ocline_model, depth_Series, depth_sigma_Series,
            d18O_Series, d18O_sigma_Series)

def d18Ocline_pymc_simple(depth, d18O, d_max=1000): # idata_fit_model, 
    '''
    Unlike d18Ocline_pymc(),
    this function is for fitting prediction d18O (climatology)

    Parameters
    ----------
    idata_fit_model : arviz.data.inference_data.InferenceData
        Return from get_thermocline_depth_thickness_relation().
    depth : Array
        Depths of d18O measurements
    d18O : Array
        Measured foraminifera d18O (permil).
    d_max : int
        Max depth (m) of the pseudo d18O profile. Somewhere between 600 and
        1000 m is appropriate.

    Returns
    -------
    idata_d18Ocline_model : arviz.InferenceData
        Inference result.

    '''

    # private variable. Scaled depth
    __depth = depth / d_max
    
    # set up pymc model
    with pm.Model() as d18Ocline_model:
        # priors from the fit_model
        # priors_fit_model = prior_from_idata.prior_from_idata(idata_fit_model, var_names=['alpha', 'beta', 'gamma', 'delta'])
        
        # Priors for unknown model parameters
        d_center_gradient = pm.Normal("d_center_gradient", mu=150, sigma=50)
        # for models it's mu=200, sigma=200
        thickness = pm.Normal("thickness", mu=150, sigma=50)
        # tilt is lognormal to prevent negative values
        tilt = pm.TruncatedNormal("tilt", mu=1.3, sigma=0.2, lower=0, upper=3)
        d18O_min = pm.Normal("d18O_min", mu=np.min(d18O), sigma=0.05)
        d18O_max = pm.Normal("d18O_max", mu=np.max(d18O), sigma=0.2)
        
        # known parameters
        __mu = d_center_gradient / d_max # scaled d center gradient
        __denom = thickness / 2.3548 / 2 / d_max * pm.math.sqrt(2)      # sigma * sqrt(2)
        # not using as sigma to y_obs because not needing to estimate sigma
        sigma = 0.1
        
        # Expected value of outcome
        mu = d18O_min + (d18O_max-d18O_min-tilt)*0.5*(1 + pm.math.erf( (__depth - __mu)/__denom)) - tilt*(1-__depth)**2 + tilt
    
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_observs", mu=mu, sigma=sigma, observed=d18O)
        
    with d18Ocline_model:
        idata_d18Ocline_model = pm.sample(progressbar=False)#target_accept=0.996) # cores=1: a bug prevents multiprocessing
        
    return idata_d18Ocline_model

def d18Ocline_pymc_schematic(depth, d18O, d_max=1000):
    '''
    Unlike d18Ocline_pymc(),
    this function is for fitting prediction d18O (climatology)

    Parameters
    ----------
    depth : Array
        Depths of d18O measurements
    d18O : Array
        Measured foraminifera d18O (permil).
    d_max : int
        Max depth (m) of the pseudo d18O profile. Somewhere between 600 and
        1000 m is appropriate.

    Returns
    -------
    idata_d18Ocline_model : arviz.InferenceData
        Inference result.

    '''

    # private variable. Scaled depth
    __depth = depth / d_max
    
    # set up pymc model
    with pm.Model() as d18Ocline_model:
     
        # Priors for unknown model parameters
        d_center_gradient = pm.Normal("d_center_gradient", mu=150, sigma=50)
        thickness = pm.Normal("thickness", mu=150, sigma=50)
        # tilt is lognormal to prevent negative values
        tilt = pm.TruncatedNormal("tilt", mu=0.3, sigma=0.2, lower=0, upper=3)
        d18O_min = pm.Normal("d18O_min", mu=np.min(d18O), sigma=0.05)
        d18O_max = pm.Normal("d18O_max", mu=np.max(d18O), sigma=0.2)
        
        # known parameters
        __mu = d_center_gradient / d_max # scaled d center gradient
        __denom = thickness / 2.3548 / 2 / d_max * pm.math.sqrt(2)      # sigma * sqrt(2)
        # not using as sigma to y_obs because not needing to estimate sigma
        sigma = 0.1
        
        # Expected value of outcome
        mu = d18O_min + (d18O_max-d18O_min-tilt)*0.5*(1 + pm.math.erf( (__depth - __mu)/__denom)) - tilt*(1-__depth)**2 + tilt
    
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_observs", mu=mu, sigma=sigma, observed=d18O)
        
    with d18Ocline_model:
        idata_d18Ocline_model = pm.sample(progressbar=False)#target_accept=0.996) # cores=1: a bug prevents multiprocessing
        
    return idata_d18Ocline_model

def species_to_ACD(species, d18O, Hol_or_LGM):
    '''
    

    Parameters
    ----------
    species : pandas.Series
        A Series of species. Each element must be among P. obliquiloculata,
        N. dutertrei, G. tumida, T. sacculifer, G. ruber.
        
    d18O : pandas.Series
        A Series of corresponding d18O measurements.

    Hol_or_LGM : Boolean
        Whether or not data is from LGM or Hol
    
    Returns
    -------
    None.

    '''
    # subsurface mean / benthic difference is inversely related to thermocline depth
    # use to adjust the three species's living depth
    subsurface_mean = np.nansum([np.nanmean(d18O[(species['Species'] == 'P. obliquiloculata')].values),
                                np.nanmean(d18O[(species['Species'] == 'N. dutertrei')].values),
                                np.nanmean(d18O[(species['Species'] == 'G. tumida')].values)]
                               ) / 3
    
    if np.isnan(subsurface_mean):
        raise ValueError('No subsurface species available for adjusting ACD')
    if Hol_or_LGM == 'Hol':
        subsurface_benthic_diff = subsurface_mean - 1.75
    elif Hol_or_LGM == 'LGM':
        # (1.75-1.34)/(617-373)*(617-505.1)+2.79 = 2.98
        subsurface_benthic_diff = subsurface_mean - 2.98
    
    species_ACD_df = pd.DataFrame(data={'Species': ['G. ruber',
                                                    'T. sacculifer', # G. was renamed T. https://doi.org/10.5194/cp-19-1359-2023
                                                    'G. sacculifer',
                                                    'P. obliquiloculata',
                                                    'N. dutertrei',
                                                    'G. tumida'],
                                        'ACD': [17, 48, 48,
                                                -60.44 * subsurface_benthic_diff - 15.33,
                                                -56.05 * subsurface_benthic_diff - 1.71,
                                                -33.5  * subsurface_benthic_diff + 76.93]})
    
    depth_df = species.merge(species_ACD_df)['ACD']
    return depth_df

def species_to_ACD_sigma(species):
    '''
    

    Parameters
    ----------
    species : pandas.Series
        A Series of species. Each element must be among P. obliquiloculata,
        N. dutertrei, G. tumida, T. sacculifer, G. ruber.

    Returns
    -------
    None.

    '''
    
    species_ACD_df = pd.DataFrame(data={'Species': ['G. ruber',
                                                    'T. sacculifer',
                                                    'G. sacculifer',
                                                    'P. obliquiloculata',
                                                    'N. dutertrei',
                                                    'G. tumida'],
                                        'ACD_1sigma': [34.5, 40, 40, 43.5, 36, 37.5]})
                                        # 'ACD_1sigma': [8.6, 10, 10, 10.88, 9, 9.38]}) # local ACD sigma should be smaller, let's say by three quarters
    
    depth_df = species.merge(species_ACD_df)['ACD_1sigma']
    return depth_df

def find_nearest_valid_profile(ds, lon, lat, var, radius=1, min_depth_coverage=700):
    '''
    This function finds the nearest profile of predicted d18O from a
    given pair of lat and lon

    Parameters
    ----------
    ds : XArray.Dataset
        Return of xr.load_dataset('Predicted_d18O.nc', decode_times=False).
    lon : double
        Site longitude.
    lat : double
        Site latitude.
    var : string
        Variable name in ds.
    radius : double
        Max radius of serach. The default is 1.
    min_depth_coverage : double
        Minimum depth (m) that the profile must have valid (non-NaN)
        data down to, so that shallow-shelf grid cells cut off by
        bathymetry aren't picked just because they're closest. The
        default is 700.

    Returns
    -------
    pandas.DataFrame
        DESCRIPTION.

    '''
    """Find the nearest valid value in a dataset for a given variable and coordinates."""
    # Select a subset of points within a certain radius
    subset = ds.sel(lon=slice(lon-radius, lon+radius), lat=slice(lat-radius, lat+radius))
    # Calculate the Euclidean distance to the points in the subset
    dist = np.sqrt((subset.lon - lon)**2 + (subset.lat - lat)**2)
    # Only keep grid cells whose profile is valid down to min_depth_coverage
    # (NaNs are a contiguous bathymetry cutoff from the bottom up, so checking
    # the depth level nearest to min_depth_coverage is enough)
    covered = ~np.isnan(subset[var].sel(depth=min_depth_coverage, method='nearest'))
    dist_covered = dist.where(covered)
    if bool(np.isnan(dist_covered.min())):
        raise ValueError(
            f"No profile found within radius {radius} of (lon={lon}, lat={lat}) "
            f"with valid data down to {min_depth_coverage} m."
        )
    # Find the coordinates of the covered point with the smallest distance
    min_dist_coords = dist_covered.where(dist_covered == dist_covered.min(), drop=True)
    nearest_lon = min_dist_coords.lon.values[0] # if there're multiple loc with equal distance, pick the first one
    nearest_lat = min_dist_coords.lat.values[0]
    # Select the nearest valid value
    nearest = subset.sel(lon=nearest_lon, lat=nearest_lat)
    return nearest[var].to_dataframe().reset_index()

def find_nearest_valid_thermocline_depth(ds, lon, lat, var, radius=1):
    '''
    This function finds the nearest thermocline depth inferred from predicted 
    d18O from a given pair of lat and lon

    Parameters
    ----------
    ds : XArray.Dataset
        Return of pd.read_excel('Predicted_d18O_pymc.xlsx') converted to a Xarray.Dataset
    lon : double
        Site longitude.
    lat : double
        Site latitude.
    var : string
        Variable name in ds.
    radius : double
        Max radius of serach. The default is 1.

    Returns
    -------
    pandas.DataFrame
        DESCRIPTION.

    '''
    """Find the nearest valid value in a dataset for a given variable and coordinates."""
    # Select a subset of points within a certain radius
    subset = ds.sel(lon=slice(lon-radius, lon+radius), lat=slice(lat-radius, lat+radius))
    if subset.lat.size == 0 or subset.lon.size == 0:
        raise ValueError('Coudn\'t find valid data within radius')
    # Calculate the Euclidean distance to the points in the subset
    dist = np.sqrt((subset.lon - lon)**2 + (subset.lat - lat)**2)
    # Create a new dataset that only includes valid values
    valid_subset = subset.where(~np.isnan(subset[var]), drop=True)
    # Find the coordinates of the valid point with the smallest distance
    min_dist_coords = dist.where(dist == dist.where(~np.isnan(valid_subset[var])).min(), drop=True)
    nearest_lon = min_dist_coords.lon.values[0] # if there're multiple loc with equal distance, pick the first one
    nearest_lat = min_dist_coords.lat.values[0]
    # Select the nearest valid value
    nearest = valid_subset.sel(lon=nearest_lon, lat=nearest_lat)
    return nearest

def isosurface(field, target, dim):
    """
    Linearly interpolate a coordinate isosurface where a field
    equals a target

    Parameters
    ----------
    field : xarray DataArray
        The field in which to interpolate the target isosurface
    target : float
        The target isosurface value
    dim : str
        The field dimension to interpolate
    depth_field: xarray DataArray
        The averaged e (z_zl) field relating potential density to depth
        
    Examples
    --------
    Calculate the depth of an isotherm with a value of 5.5:
    
    >>> temp = xr.DataArray(
    ...     range(10,0,-1),
    ...     coords={"depth": range(10)}
    ... )
    >>> isosurface(temp, 5.5, dim="depth")
    <xarray.DataArray ()>
    array(4.5)
    """
    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = field.isel(slice0).drop(dim)
    field1 = field.isel(slice1).drop(dim)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(
        crossing_mask_decr | crossing_mask_incr, 1, np.nan
    )

    coords0 = crossing_mask * field[dim].isel(slice0).drop(dim)
    coords1 = crossing_mask * field[dim].isel(slice1).drop(dim)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = (
        coords0 + (target - field0) * 
        (coords1 - coords0) / (field1 - field0)
    )

    return iso.max(dim, skipna=True)

