import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

def create_task_confounders_old(frame_times, events_df, fir_delays):
    
    trial_types = events_df.trial_type.unique()
    task_conf_reg = []
    for trial_name in trial_types:
        cond = events_df.trial_type==trial_name
        trial_events = events_df.loc[cond, ["onset", "duration"]].to_numpy()
        trial_events = np.column_stack((trial_events, np.ones(trial_events.shape[0]))).T # Add amplitudes
        trial_events_reg, _ = compute_regressor(trial_events, hrf_model="fir", fir_delays=fir_delays,frame_times=frame_times)
        task_conf_reg.append(trial_events_reg)
        
    return np.column_stack(task_conf_reg)

def create_task_confounders(frame_times, events_df, hrf_model, fir_delays):
    
    trial_types = events_df.trial_type.unique()
    task_conf_reg = []
    for trial_name in trial_types:
        cond = events_df.trial_type==trial_name
        trial_events = events_df.loc[cond, ["onset", "duration"]].to_numpy()
        trial_events = np.column_stack((trial_events, np.ones(trial_events.shape[0]))).T # Add amplitudes
        trial_events_reg, _ = compute_regressor(trial_events, 
                                                hrf_model=hrf_model, 
                                                fir_delays=fir_delays, 
                                                frame_times=frame_times)
        task_conf_reg.append(trial_events_reg)
        
    return np.column_stack(task_conf_reg)

def denoise(X, Y):
    
    # Compute mean data
    X_offset = np.mean(X, axis=0)
    Y_offset = np.mean(Y, axis=0)
     
    # Compute coefficients
    beta, _, _, _ = np.linalg.lstsq(X - X_offset, Y-Y_offset, rcond=None)
    
    Y_clean = Y - X @ beta 
    
    return Y_clean

def denoise_w_task(X, Y, T):
    
    # Compute mean data
    X_offset = np.mean(X, axis=0)
    Y_offset = np.mean(Y, axis=0)
    T_offset = np.mean(T, axis=0)
     

    # Compute coefficients
    beta, _, _, _ = np.linalg.lstsq(np.column_stack((X - X_offset, T - T_offset)), Y-Y_offset, rcond=None)
    
    Y_clean = Y - X @ beta[:X.shape[1],:]
    
    return Y_clean

def standardize(X):
    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < np.finfo(np.float).eps] = 1.  # avoid numerical problems
    
    Z = (X - mu)/std
    
    return Z

def band_pass_dct(high_pass, low_pass, frame_times):
    """Create a cosine drift matrix with periods greater or equals to period_cut
    Parameters
    ----------
    period_cut: float
         Cut period of the low-pass filter (in sec)
    frametimes: array of shape(nscans)
         The sampling times (in sec)
    Returns
    -------
    cdrift:  array of shape(n_scans, n_drifts)
             cosin drifts plus a constant regressor at cdrift[:,0]
    Ref: http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II
    """

    if (high_pass is None) & (low_pass is None):
        raise(ValueError("No frequency passed"))

    if (high_pass is not None) & (low_pass is not None):
        if low_pass < high_pass: 
            raise(ValueError("Low pass has to be greater than high pass"))

    # frametimes.max() should be (len_tim-1)*dt
    dt = frame_times[1] - frame_times[0]
    f_max = 0.5*(1/dt) # nyquist frequency

    if low_pass:
        if low_pass > f_max:
            low_pass = None # this means it just perform a high-pass filtering

    len_tim = len(frame_times)
    n_times = np.arange(len_tim)

    k_max = int(len_tim)
#    print("k_max: ", k_max)

    hp_order,lp_order = [],[]
    if high_pass:
        k_hp = max(int(np.floor(2 * len_tim * high_pass * dt)), 1)
 #       print("k_hp: ", k_hp)
        hp_order = np.arange(1, k_hp + 1) # This +1 is to have the same as in nilearn https://github.com/nilearn/nilearn/blob/1607b52458c28953a87bbe6f42448b7b4e30a72f/nilearn/glm/first_level/design_matrix.py#L80
    if low_pass:
        k_lp = max(int(np.ceil(2 * len_tim * low_pass * dt)), 1)
  #      print("k_lp: ", k_lp)
        lp_order = np.arange(k_lp-1, k_max) # This -1 does not affect for our study

    orders = np.concatenate((hp_order, lp_order))
    # hfcut = 1/(2*dt) yields len_time
    # If series is too short, return constant regressor
    #order = max(int(np.floor(2 * len_tim * hfcut * dt)), 1)
    cdrift = np.zeros((len_tim, len(orders)))
    nfct = np.sqrt(2.0 / len_tim)

    for ii, k in enumerate(orders):
        cdrift[:, ii] = nfct * np.cos((np.pi / len_tim) * (n_times + 0.5) * k)

    #cdrift[:, order - 1] = 1.0  # or 1./sqrt(len_tim) to normalize
    return cdrift

def ar_whiten(X):
    """
    Whiten time series matrix using an AR(1) model

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statsmodels.regression.linear_model import yule_walker

    np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
    # Demean time series. This would be just removing the intercept from
    # the denoised time series
    
    X = X - X.mean(0)

    ar_coefs, _ = np.apply_along_axis(yule_walker, 0, X, demean=False)
    ar_coefs = np.concatenate(ar_coefs)

    whitened_X = X.copy()

    whitened_X[1:] = whitened_X[1:] - ar_coefs * X[0:-1]

    return whitened_X
