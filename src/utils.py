import pandas as pd
import numpy as np
from nilearn import glm

def load_config_file(yaml_file):
    import yaml  

    from fractions import Fraction

    def fraction_constructor(loader, node):
        value = loader.construct_scalar(node)
        return float(Fraction(value))

    yaml.add_constructor('!fraction', fraction_constructor)

    with open(yaml_file, 'r') as file:   
        config_dat = yaml.load(file, Loader=yaml.FullLoader)
    
    pass_tests = True
    keys = ['edge_denoise_opts', 'hrf_model', 
            'noise_model', 'high_pass', 
            'drift_model', 'standardize', 
            'gsr']
    
    for key in keys:
        if key not in list(config_dat.keys()):
            pass_tests = False
            
    if pass_tests:
        if "denoise_task" not in list(config_dat['edge_denoise_opts'].keys()):
            pass_tests = False
            
    assert pass_tests
    
    # Convert any "None" to None
    for key, value in config_dat.items():
        if value == "None":
            config_dat[key] = None
        
    for key, value in config_dat['edge_denoise_opts'].items():
        if value == "None":
            config_dat['edge_denoise_opts'][key] = None
            
    # If hrf model in edge_denoise_opts is equal to "fir", set the delays to account for a modelling response of up to 24 secs 
    if config_dat['edge_denoise_opts']['hrf_model'] == 'fir':
    	config_dat['edge_denoise_opts']['fir_delays'] = list(range(1, 13))
    else:
    	config_dat['edge_denoise_opts']['fir_delays'] = [0]
    return config_dat

def create_case_outdir(case_opts):

    output_dir = "pipeline"
    output_dir += "_events-" + str(int(case_opts['edge_denoise_opts']['denoise_task']))

    if case_opts['edge_denoise_opts']['hrf_model'] == case_opts['hrf_model']:
        output_dir += "_denoise-" + "Node"
    else:
        output_dir += "_denoise-" + case_opts['edge_denoise_opts']['hrf_model']

    output_dir += "_whiten-" + case_opts['noise_model']

    if case_opts['hrf_model'] is None:
        output_dir += "_hrf-0"
    else:
        output_dir += "_hrf-1"

    output_dir += "_zscore-" + str(int(case_opts['standardize']))

    output_dir += "_gsr-" + str(int(case_opts['gsr']))

    return output_dir

def compute_task_regressors(events_file):

    """

    Function to generate the matrix of task regressors including the
    convolved events, their derivatives and dispersions

    """
    events_df = pd.read_csv(events_file, sep="\t")
    
    cond = events_df.trial_type=="Congruent"
    cong_cond = events_df.loc[cond, ["onset", "duration"]].to_numpy()
    cong_cond = np.column_stack((cong_cond, np.ones(cong_cond.shape[0]))).T
    cong_cond = glm.first_level.compute_regressor(cong_cond,
                                                  hrf_model="glover + derivative + dispersion",
                                                  frame_times=np.arange(280)*2.0)
    
    cond = events_df.trial_type=="Incongruent"
    incg_cond = events_df.loc[cond, ["onset", "duration"]].to_numpy()
    incg_cond = np.column_stack((incg_cond, np.ones(incg_cond.shape[0]))).T
    incg_cond = glm.first_level.compute_regressor(incg_cond,
                                                  hrf_model="glover + derivative + dispersion",
                                                  frame_times=np.arange(280)*2.0)
        
    return np.column_stack((cong_cond[0], incg_cond[0]))

def compute_task_regressors_fir(events_file, fir_delays):

    events_df = pd.read_csv(events_file, sep="\t")
    
    cond = events_df.trial_type=="Congruent"
    cong_cond = events_df.loc[cond, ["onset", "duration"]].to_numpy()
    cong_cond = np.column_stack((cong_cond, np.ones(cong_cond.shape[0]))).T
    cong_cond = glm.first_level.compute_regressor(cong_cond,
                                                  hrf_model="fir",
                                                  fir_delays = fir_delays,
                                                  frame_times=np.arange(280)*2.0)
    
    cond = events_df.trial_type=="Incongruent"
    incg_cond = events_df.loc[cond, ["onset", "duration"]].to_numpy()
    incg_cond = np.column_stack((incg_cond, np.ones(incg_cond.shape[0]))).T
    incg_cond = glm.first_level.compute_regressor(incg_cond,
                                                  hrf_model="fir",
                                                  fir_delays=fir_delays,
                                                  frame_times=np.arange(280)*2.0)
        
    return np.column_stack((cong_cond[0], incg_cond[0]))

def compute_edge_ts(roi_mat):
    
    """
    
    Function to compute the unwarped time
    from a matrix of time series
    
    """
    n_rois = roi_mat.shape[1]
    n_vols = roi_mat.shape[0]
    
    #n_edges = int(n_rois*(n_rois-1)/2)
    
    edge_mat = np.zeros((n_rois, n_rois, 1, n_vols ))
    
    for ii in range(n_rois):
        for jj in range(ii+1, n_rois):
            edge_mat[ii, jj, 0, :] = roi_mat[:,ii]*roi_mat[:,jj]
            edge_mat[jj, ii, 0,:] = edge_mat[ii, jj, 0, :]
            
    return edge_mat

def create_edge_mask_from_atlas(atlas_file):
    """
    Function to create a mask for the edge 
    time series, so as to take only the upper
    diagonal terms.
    
    """ 
    
    from nilearn import image
    
    atlas_img = image.load_img(atlas_file)
    n_parcels = int(len(np.unique(atlas_img.get_fdata()))-1) # to discard background 0
    
    mask_data = np.zeros(shape=(n_parcels, n_parcels), dtype=int)
    mask_data[np.triu_indices_from(mask_data, k=1)]=1
    mask_data = mask_data[:,:,None]
    mask_img = image.new_img_like(ref_niimg = atlas_img, 
                                  data = mask_data, 
                                  affine = np.eye(4))
    
    return mask_img
    
def t_to_r(t, df):
    """
    Function to convert a t-statistic to a pearson correlation
    https://sscc.nimh.nih.gov/sscc/gangc/tr.html
    """
    sign = np.sign(t)
    rsquare = t**2/(t**2 + df)
    return sign*np.sqrt(rsquare)

def map_on_atlas(stat_data, atlas_img):
    """
    Function to map a statistical map
    """
    from nilearn import image
    atlas_img = image.load_img(atlas_img)
    n_rois = len(np.unique(atlas_img.get_fdata()))-1
    stat_data = np.squeeze(stat_data)
    if stat_data.ndim > 1:
        raise ValueError("input stat data should be a vector")
    if len(stat_data) != n_rois:
        raise ValueError("stat componentes do not match"
                         " the number of ROIS of the atlas")
    atlas_img_data = atlas_img.get_fdata()
    stat_on_atlas = np.zeros_like(atlas_img_data)
    
    for ii in range(n_rois):
        stat_on_atlas[atlas_img_data==(ii+1)] = stat_data[int(ii)]
    
    stat_on_atlas_img = image.new_img_like(atlas_img, stat_on_atlas)
    return stat_on_atlas_img
