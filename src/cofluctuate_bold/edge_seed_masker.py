# The only thing that chages between both is 

from nilearn.input_data import NiftiSpheresMasker, NiftiMasker
from nilearn.image import load_img, new_img_like
import numpy as np
import pandas as pd
import os

from .utils import create_task_confounders, denoise_task, standardize

class NiftiEdgeSeed():

    def __init__(self,
                 seed,
                 radius = None,
                 mask_img = None,
                 smoothing_fwhm = None,
                 detrend = False,
                 low_pass = None,
                 high_pass = None, 
                 t_r = None,
                 fir_delays=[0]
                 ):

        self.seed = seed
        self.radius = radius
        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.fir_delays = fir_delays

    def fit_transform(self,
                      run_img,
                      events=None,
                      confounds=None):

        run_img = load_img(run_img)
        n_scans = run_img.shape[3]

        #TODO: See if it makes sense to create a function for this
        # or a base class that has this method

        # 1- Load and compute FIR events
        task_conf = None
        if events is not None:
            if isinstance(events, str):
                assert os.path.exists(events)
                assert events.endswith("events.tsv")
                events_mat = pd.read_csv(events, sep="\t")

                start_time = 0
                end_time = (n_scans - 1)* self.t_r
                frame_times = np.linspace(start_time, end_time, n_scans)
                task_conf = create_task_confounders(frame_times, events_mat,
                                                    fir_delays = self.fir_delays)

            elif isinstance(events, np.ndarray):
                # You can supply a given task matrix to denoise
                task_conf = events

        self.task_conf_ = task_conf

        # 2- Get seed region and clean it
        seed_masker = NiftiSpheresMasker(seeds = [self.seed],
                                         radius = self.radius,
                                         detrend = self.detrend,
                                         low_pass = self.low_pass,
                                         high_pass = self.high_pass,
                                         t_r = self.t_r,
                                         standardize = False)
        seed_ts_conf = seed_masker.fit_transform(run_img, confounds=confounds)
        self.seed_ts_conf_ = seed_ts_conf.copy()

        if events is not None:
            seed_ts_conf_task = denoise_task(X = task_conf, Y = seed_ts_conf)
        else:
            seed_ts_conf_task = seed_ts_conf
        self.seed_ts_conf_task_ = seed_ts_conf_task.copy()

        seed_ts_zscore = standardize(seed_ts_conf_task)

        # 2- Get voxel data from a brain mask
        brain_mask = NiftiMasker(mask_img = self.mask_img,
                                 smoothing_fwhm = self.smoothing_fwhm,
                                 detrend = self.detrend,
                                 low_pass = self.low_pass,
                                 high_pass = self.high_pass,
                                 t_r = self.t_r,
                                 standardize = False)
        brain_ts_conf = brain_mask.fit_transform(run_img, confounds=confounds)
        self.brain_ts_conf_ = brain_ts_conf.copy()

        if events is not None:
            brain_ts_conf_task = denoise_task(X = task_conf, Y = brain_ts_conf)
        else:
            brain_ts_conf_task = brain_ts_conf
        self.brain_ts_conf_task_ = brain_ts_conf_task.copy()

        brain_ts_zscore = standardize(brain_ts_conf_task)

        # 3- Multiply seed region with brain
        edge_ts = brain_ts_zscore*seed_ts_zscore
        # Resulting data back to img space
        edge_ts_img = brain_mask.inverse_transform(edge_ts)
        return edge_ts_img
