def edge_img_first_level(run_img, 
                         events_file, 
                         conf_df, 
                         atlas_file, 
                         denoise_opts, 
                         first_level_opts,
                         mask_img = None,                        
                         intercept_only=False):
    
    """
    
    Function to compute a first level object for
    the edge time seris from a given bold image,
    with confounders and an events file.
    
    """
    from nilearn.glm.first_level import FirstLevelModel
    from cofluctuate_bold_glm import NiftiEdgeAtlas
    import pandas as pd

    # Compute edge time series
    edge_atlas = NiftiEdgeAtlas(atlas_file = atlas_file, **denoise_opts)
    edge_atlas_img = edge_atlas.fit_transform(run_img=run_img, events = events_file, confounds=conf_df)
    
    # Compute first level for this
    fmri_glm = FirstLevelModel(mask_img=mask_img, **first_level_opts)
    
    if intercept_only:
        design_matrix = pd.DataFrame({'constant': [1]*edge_atlas_img.shape[3]}) # Just a constant, we don't have events here
        fmri_glm.fit(run_imgs=edge_atlas_img, design_matrices=design_matrix)
    else:
        fmri_glm.fit(run_imgs=edge_atlas_img, events = events_file, confounds=None)
        
    return fmri_glm

def get_contrasts(intercept_only):
    """
    
    Function that returns the contrasts
    to run.
    
    """
    
    if intercept_only:
        contrasts = ["constant"]
    else:
        contrasts = ["constant", "Congruent", "Incongruent", "Incongruent-Congruent", "Congruent-constant", "Incongruent-constant"]
    return contrasts
