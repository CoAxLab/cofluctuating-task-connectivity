#!/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from os.path import join as opj

import shutil
import tempfile
import gc
from joblib import delayed, Parallel

from scipy.spatial.distance import squareform
from nilearn.glm.first_level import compute_regressor
from nilearn.image import load_img

def load_data(task_id):
    """
    Function to load the edge imgs and the motion outliers as covariates in
    estimating the group-level effect sizes.
    """

    base_dir = opj(project_dir, "results/edge_imgs_gsr/shen/task-%s" % task_id)
    filename = "sub-%d" + "_ses-01_task-%s_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz" % task_id

    pattern = opj(base_dir, filename)

    edge_imgs_2d = []
    for subj in tqdm(final_subjects):
        edge_img = load_img(pattern % subj)
        edge_img_data = np.squeeze(edge_img.get_fdata()) # get data and drop third dummy dimensions
        edge_data_2d = np.row_stack([squareform(edge_img_data[:,:,ii], checks=False) for ii in range(n_scans)])
        edge_imgs_2d.append(edge_data_2d)

    return edge_imgs_2d


def estimate_model(idxs, edge_imgs_2d):

    from sklearn.linear_model import LinearRegression
    import numpy as np

    linReg = LinearRegression()

    first_level_betas = []
    first_level_intercepts = []

    for ix in idxs:
        # Edge time series
        Y = edge_imgs_2d[ix,:,:]

        linReg.fit(X, Y)
        first_level_betas.append(linReg.coef_)
        first_level_intercepts.append(linReg.intercept_)

    betas_avg = np.array(first_level_betas).mean(axis=0)
    intercepts_avg = np.array(first_level_intercepts).mean(axis=0)

    return intercepts_avg, betas_avg

def prediction(intercepts_avg, betas_avg):
    return intercepts_avg + X.dot(betas_avg.T)

def _compute_score_cv(within_task,
                      between_task,
                      n_splits,
                      seed):

    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros(shape=(n_subjects, n_subjects))

    n_links = within_task.shape[-1]

    rsquare_within = np.zeros(shape=(n_splits, n_links))
    pearson_within = np.zeros(shape=(n_splits, n_links))

    rsquare_between = np.zeros(shape=(n_splits, n_links))
    pearson_between = np.zeros(shape=(n_splits, n_links))

    for i_split, (train_idxs, test_idxs) in enumerate(cv.split(dummy_X)):
        # Estimate model
        intercepts_avg, betas_avg = estimate_model(train_idxs, within_task)

        # Prediction using this estimation
        Y_pred = prediction(intercepts_avg, betas_avg)

        # test
        Y_test_within = within_task[test_idxs,:,:].mean(0)
        Y_test_between = between_task[test_idxs,:,:].mean(0)

        rsquare_within[i_split,:] = [r2_score(Y_test_within[:,ii], Y_pred[:,ii]) for ii in range(n_links)]
        pearson_within[i_split,:] = [np.corrcoef(Y_test_within[:,ii], Y_pred[:,ii])[0,1] for ii in range(n_links)]

        rsquare_between[i_split,:] = [r2_score(Y_test_between[:,ii], Y_pred[:,ii]) for ii in range(n_links)]
        pearson_between[i_split,:] = [np.corrcoef(Y_test_between[:,ii], Y_pred[:,ii])[0,1] for ii in range(n_links)]

    return rsquare_within, pearson_within, rsquare_between, pearson_between


#################################################################################
################### 0-Some definitions for the scrpit ##########################

project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"

final_subjects = np.loadtxt(opj(project_dir, "data/subjects_intersect_motion_035.txt"))
n_subjects = len(final_subjects)
print("the number of subjects is", n_subjects)

n_scans = 280
print("the number of scans is ", n_scans)

t_r = 2.0
print("Repetition time", t_r)

frame_times = np.arange(n_scans)*t_r

n_splits = 3 # 3-Fold CV for out-of-sample performance
n_shuffles = 20 # Repeat this 20 times
n_jobs = 10 # Number of parallel jobs that will run a CV each

#################################################################################
################### 1-Create input matrix of task effects #######################

print("Creating input matrix of task effects...")

task_events = pd.read_csv(opj(project_dir, "data/task-stroop_events.tsv"), sep="\t")

inc_cond = task_events[task_events.loc[:,"trial_type"]=="Incongruent"].to_numpy().T
inc_cond[2,:] = [1,1,1,1]
con_cond = task_events[task_events.loc[:,"trial_type"]=="Congruent"].to_numpy().T
con_cond[2,:] = [1,1,1,1]

inc_regressor, _ = compute_regressor(inc_cond,
                                     hrf_model = "glover + derivative + dispersion",
                                     frame_times=frame_times)
inc_regressor = np.squeeze(inc_regressor)

con_regressor, _ = compute_regressor(con_cond,
                                     hrf_model = "glover + derivative + dispersion",
                                     frame_times=frame_times)
con_regressor = np.squeeze(con_regressor)

X = np.column_stack((inc_regressor, con_regressor))

#output_dir = Path(opj(project_dir, "results/generalizability/gsr"))
output_dir = Path(opj(project_dir, "results/predictions/gsr"))
output_dir.mkdir(exist_ok=True, parents=True)
output_dir = output_dir.resolve().as_posix()

#################################################################################
################### 2-Loading edge imgs and motion ouliers ######################
print("Loading data...")

temp_dir = tempfile.mkdtemp()

edges_stroop = load_data(task_id="stroop")
edges_stroop = np.array(edges_stroop)
# Convert edges data to memmap array, in order not to blow RAM memory
edges_stroop_mem = np.memmap(temp_dir + "/" + "edges_stroop.npy",
                             dtype = edges_stroop.dtype,
                             shape = edges_stroop.shape,
                             mode='w+')
edges_stroop_mem[:] = edges_stroop[:]
del edges_stroop

edges_msit = load_data(task_id="msit")
edges_msit = np.array(edges_msit)
# Convert edges data to memmap array, in order not to blow RAM memory
edges_msit_mem = np.memmap(temp_dir + "/" + "edges_msit..npy",
                             dtype = edges_msit.dtype,
                             shape = edges_msit.shape,
                             mode='w+')
edges_msit_mem[:] =  edges_msit[:]
del edges_msit


##################################################################################
################### 3-Within and between tasks performance #######################

# First, using stroop as training test
print("Stroop as training...")

parallel = Parallel(n_jobs=n_jobs)
res = parallel(delayed(_compute_score_cv)(within_task = edges_stroop_mem,
                                          between_task =  edges_msit_mem,
                                          n_splits = n_splits,
                                          seed= seed)
               for seed in tqdm(range(n_shuffles))
              )

r2_stroop_stroop, pearson_stroop_stroop,  r2_stroop_msit, pearson_stroop_msit = zip(*res)

np.savez(opj(output_dir, "scores_stroop_stroop.npz"),
         r2_scores = np.array(r2_stroop_stroop),
         pearson_scores = np.array(pearson_stroop_stroop)
        )

np.savez(opj(output_dir, "scores_stroop_msit.npz"),
         r2_scores = np.array(r2_stroop_msit),
         pearson_scores = np.array(pearson_stroop_msit)
        )


del parallel
_ = gc.collect()

# Second, using MSIT as training
print("MSIT as training...")

parallel = Parallel(n_jobs=n_jobs)
res = parallel(delayed(_compute_score_cv)(within_task = edges_msit_mem,
                                          between_task =  edges_stroop_mem,
                                          n_splits = n_splits,
                                          seed= seed)
               for seed in tqdm(range(n_shuffles))
              )

r2_msit_msit, pearson_msit_msit, r2_msit_stroop, pearson_msit_stroop = zip(*res)

np.savez(opj(output_dir, "scores_msit_msit.npz"),
         r2_scores = np.array(r2_msit_msit),
         pearson_scores = np.array(pearson_msit_msit)
        )

np.savez(opj(output_dir, "scores_msit_stroop.npz"),
         r2_scores = np.array(r2_msit_stroop),
         pearson_scores = np.array(pearson_msit_stroop)
        )


del parallel
_ = gc.collect()

shutil.rmtree(temp_dir)

