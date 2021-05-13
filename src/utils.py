import pandas as pd
import numpy as np
from nilearn import glm

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
