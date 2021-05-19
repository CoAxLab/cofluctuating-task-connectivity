import os
import numpy as np
import pandas as pd

from nilearn.input_data import NiftiSpheresMasker, NiftiMasker
from nilearn.image import load_img

from sklearn.utils import check_array

from .utils import create_task_confounders, denoise, standardize, band_pass_dct

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
                 fir_delays=[0]):

        self.seed = seed
        self.radius = radius
        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.fir_delays = fir_delays

    def fit(self):
        """Fit function for scikit-compatibility. it pnly carries checks on the input params"""

        return self

    def transform(self,
                  run_img,
                  events=None,
                  confounds=None):

        run_img = load_img(run_img)
        n_scans = run_img.shape[3]
        start_time = 0
        end_time = (n_scans - 1)*self.t_r
        frame_times = np.linspace(start_time, end_time, n_scans)
        #TODO: See if it makes sense to create a function for this
        # or a base class that has this method

        # 1- Get seed region
        seed_masker = NiftiSpheresMasker(seeds=[self.seed],
                                         radius=self.radius,
                                         detrend=None,
                                         low_pass=None,
                                         high_pass=None,
                                         t_r=self.t_r,
                                         standardize=False)
        seed_ts = seed_masker.fit_transform(run_img)

        # 2- Get voxel data from a brain mask
        brain_mask = NiftiMasker(mask_img=self.mask_img,
                                 smoothing_fwhm=self.smoothing_fwhm,
                                 detrend=None,
                                 low_pass=None,
                                 high_pass=None,
                                 t_r=self.t_r,
                                 standardize=False)

        brain_ts = brain_mask.fit_transform(run_img)

        # 3-Load and compute FIR events
        task_conf = None
        if events is not None:
            if isinstance(events, str):
                assert os.path.exists(events)
                assert events.endswith("events.tsv")
                events_mat = pd.read_csv(events, sep="\t")

                task_conf = create_task_confounders(frame_times, events_mat,
                                                    fir_delays=self.fir_delays)
            else:
                # You can supply a given task matrix to denoise
                task_conf = check_array(events)
        else:
            task_conf = np.array([]).reshape(n_scans, 0)

        # 3-Create matrix of drifts
        if (self.high_pass is not None) | (self.low_pass is not None):
            drifts_mat = band_pass_dct(high_pass = self.high_pass,
                                       low_pass = self.low_pass,
                                       frame_times = frame_times)
        else:
            drifts_mat = np.array([]).reshape(n_scans, 0)

        # 4- Create other confounders matrix
        if confounds is None:
            conf_mat = np.array([]).reshape(n_scans, 0)
        else:
            conf_mat = check_array(confounds)

        # 5-Create denoising matrix
        denoise_mat = np.column_stack((task_conf, conf_mat, drifts_mat))
        self.denoise_mat_ = denoise_mat

        if denoise_mat.shape[1] > 0:
            seed_ts_denoised = denoise(X=denoise_mat, Y = seed_ts)
            brain_ts_denoised = denoise(X=denoise_mat, Y = brain_ts)
        else:
            seed_ts_denoised = seed_ts
            brain_ts_denoised = brain_ts

        self.seed_ts_denoised_ = seed_ts_denoised.copy()
        self.brain_ts_denoised_ = brain_ts_denoised.copy()

        # 6-Standardize both objects
        seed_ts_denoised = standardize(seed_ts_denoised)
        brain_ts_denoised = standardize(brain_ts_denoised)

        # 7- Multiply seed region with brain
        edge_ts = brain_ts_denoised*brain_ts_denoised
        edge_img = brain_mask.inverse_transform(edge_ts) # Resulting data back to img space
        return edge_img

    def fit_transform(self,
                      run_img,
                      events = None,
                      confounds = None):

            return self.fit().transform(run_img,
                                        events = events,
                                        confounds = confounds)
