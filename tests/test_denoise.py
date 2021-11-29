def test_seed_denoise():
    import numpy as np
    from os.path import join as opj
    from sklearn.preprocessing import StandardScaler
    base_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
    final_subjects = np.loadtxt(opj(base_dir,
                                    "data",
                                    "subjects_intersect_motion_035.txt"))
    pattern_seed = opj(base_dir,
                    "results/edge_imgs_gsr/seed",
                    "task-stroop",
                    "positive",
                    "denoised_seed_time_series",
                    "sub-%d_ses-01_task-stroop_space-MNI152NLin2009cAsym_desc-conf_seed.npy")
    pattern_denoise = opj(base_dir,
                         "results/edge_imgs_gsr/seed",
                         "task-stroop",
                         "positive",
                         "denoising_mats",
                         "sub-%d_ses-01_task-stroop_space-MNI152NLin2009cAsym_desc-denoise_mat.npy")

    ss = StandardScaler()
    eps = 1e-10
    for subj in final_subjects:
        seed_ts = np.load(pattern_seed % subj)
        seed_ts_z = ss.fit_transform(seed_ts)

        denoise_mat = np.load(pattern_denoise % subj)
        denoise_mat_z = ss.fit_transform(denoise_mat)

        cors = np.dot(denoise_mat_z.T, seed_ts_z)
        max_cor = abs(cors).max()
        print(max_cor)
        assert max_cor < eps
