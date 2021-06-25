import os
import numpy as np
import pandas as pd

from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img, new_img_like

from sklearn.utils import check_array

from .utils import create_task_confounders, denoise, standardize, band_pass_dct


def compute_edge_ts(roi_mat):

    """
    Function to compute the unwarped time
    from a matrix of time series

    """
    n_rois = roi_mat.shape[1]
    n_vols = roi_mat.shape[0]

    edge_mat = np.zeros((n_rois, n_rois, 1, n_vols ))

    for ii in range(n_rois):
        for jj in range(ii+1, n_rois):
            edge_mat[ii, jj, 0, :] = roi_mat[:,ii]*roi_mat[:,jj]
            edge_mat[jj, ii, 0,:] = edge_mat[ii, jj, 0, :]

    return edge_mat


class NiftiEdgeAtlas():

    def __init__(self,
                 atlas_file,
                 detrend = False,
                 low_pass = None,
                 high_pass= None,
                 t_r = None,
                 hrf_model = "fir",
                 fir_delays=[0]
                ):

        self.atlas_file = atlas_file
        self.detrend = detrend
        self.low_pass = None # For the moment and this project, only high_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.hrf_model = hrf_model
        self.fir_delays = fir_delays

    def fit(self):
        """Fit function for scikit-compatibility. it pnly carries checks on the input params"""

        return self


    def transform(self,
                  run_img,
                  events = None,
                  confounds = None):

        run_img = load_img(run_img)
        n_scans = run_img.shape[3]
        start_time = 0
        end_time = (n_scans - 1)* self.t_r
        frame_times = np.linspace(start_time, end_time, n_scans)

        # 1- Parcellate data
        label_masker = NiftiLabelsMasker(labels_img = self.atlas_file,
                                         detrend = None,
                                         low_pass = None,
                                         high_pass = None,
                                         t_r = self.t_r,
                                         standardize=False)
        atlas_roi_ts = label_masker.fit_transform(run_img)

        #TODO: See if it makes sense to create a function for this
        # or a base class that has this method

        # 2-Load and compute task confounders matrix
        task_conf = None
        if events is not None:
            if isinstance(events, str):
                assert os.path.exists(events)
                assert events.endswith("events.tsv")
                events_mat = pd.read_csv(events, sep="\t")

                task_conf = create_task_confounders(frame_times, events_mat,
                                                    hrf_model=self.hrf_model,
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
            atlas_roi_ts_denoised = denoise(X=denoise_mat, Y = atlas_roi_ts)
        else:
            atlas_roi_ts_denoised = atlas_roi_ts

        self.atlas_roi_denoised_ = atlas_roi_ts_denoised

        # 6-Standardize data
        atlas_ts_clean =  standardize(atlas_roi_ts_denoised)

        edge_ts = compute_edge_ts(atlas_ts_clean)
         # Create new image, adding fake affine (old was:run_img.affine)
        edge_img = new_img_like(run_img, edge_ts, affine = np.eye(4))

        return edge_img

    def fit_transform(self,
                      run_img,
                      events = None,
                      confounds = None):

        return self.fit().transform(run_img,
                                    events = events,
                                    confounds = confounds)
