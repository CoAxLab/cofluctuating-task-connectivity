########## TASK/RESTING DENOISE OPTIONS #################
# For resting, we are not going to pass any events file #
# so fir delays will have no effect                     #
#########################################################

_T_R = 2.0
_HIGH_PASS = 1/187.
_DETREND = False
_FIR_DELAYS = list(range(1, 13)) # NEW: Modelling response up to 24 secs #list(range(1,10))
_HRF_MODEL_EDGE_DENOISE = "fir"
def get_denoise_opts():
    
    denoise_dict = dict()
    denoise_dict['t_r'] = _T_R
    denoise_dict['high_pass'] = _HIGH_PASS
    denoise_dict['detrend'] = _DETREND
    denoise_dict['fir_delays'] = _FIR_DELAYS
    denoise_dict['hrf_model'] = _HRF_MODEL_EDGE_DENOISE
    print("getting denoise options: ", denoise_dict)
    
    return denoise_dict

################ FIRST-LEVEL EDGE OPTIONS ###############
#                                                       #
#                                                       #
#########################################################

_HRF_MODEL_EDGE = "glover + derivative + dispersion"
_DRIFT_MODEL_EDGE = None
_SMOOTHING_FWHM_EDGE = None
_SIGNAL_SCALING_EDGE = False

def get_first_level_edge_opts():
    
    first_level_dict = dict()
    
    first_level_dict['t_r'] = _T_R
    first_level_dict['hrf_model'] = _HRF_MODEL_EDGE
    first_level_dict['smoothing_fwhm'] = _SMOOTHING_FWHM_EDGE
    first_level_dict['drift_model'] = _DRIFT_MODEL_EDGE
    first_level_dict['signal_scaling'] = _SIGNAL_SCALING_EDGE
    
    print("getting first level options: ", first_level_dict)
    
    return first_level_dict
    
################ FIRST-LEVEL NODE OPTIONS ##############
#                                                       #
#                                                       #
#########################################################

_HRF_MODEL_NODE = "glover + derivative + dispersion"
_DRIFT_MODEL_NODE = 'cosine'
_SMOOTHING_FWHM_NODE = None

def get_first_level_node_opts():
    
    first_level_dict = dict()
    
    first_level_dict['t_r'] = _T_R
    first_level_dict['hrf_model'] = _HRF_MODEL_NODE
    first_level_dict['smoothing_fwhm'] = _SMOOTHING_FWHM_NODE
    first_level_dict['drift_model'] = _DRIFT_MODEL_NODE
    first_level_dict['high_pass'] = _HIGH_PASS
    
    print("getting first level options: ", first_level_dict)
    
    return first_level_dict
