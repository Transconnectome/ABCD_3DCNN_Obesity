import copy
import numpy as np 
import pandas as pd

import glob 
import matplotlib.pyplot as plt
import os 

import nibabel as nib

from monai.transforms import Compose, AddChannel, Resize 
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img 
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report 


import argparse

import warnings
warnings.filterwarnings("ignore")

## arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--target",default='sex',type=str, required=False,help='')
parser.add_argument("--study_sample",default='after1y',type=str, choices=['after1y', 'after2y'],required=False,help='')
parser.add_argument("--time_point",default='baseline_year_1_arm_1',type=str, choices=['baseline_year_1_arm_1', '1_year_follow_up_y_arm_1', '2_year_follow_up_y_arm_1'],required=False,help='')
parser.add_argument("--smoothing_fwhm",default=0.0,type=float,required=False,help='')
# arguments for parameteric test
parser.add_argument("--alpha",default=0.05,type=float,required=False,help='')
parser.add_argument("--correction",default='fdr',type=str, choices=['fpr', 'fdr', 'bonferroni'],required=False,help='')
# arguments for non-parametric (permutation) test 
parser.add_argument("--permutation_test",action='store_true',required=False,help='')
parser.set_defaults(permutation_test = False)
parser.add_argument("--n_perm",default=1000,type=int, required=False,help='')
parser.add_argument("--threshold",default=1,type=float,required=False,help='Since we are plotting negative log p-values and using a threshold equal to 1, it corresponds to corrected p-values lower than 10%, meaning that there is less than 10% probability to make a single false discovery (90% chance that we make no false discoveries at all).')
parser.add_argument("--tfce",action='store_true',required=False,help='')
parser.set_defaults(tfce = False)
parser.add_argument("--n_process",default=-1,type=int,required=False,help='')
# arguments for others 
parser.add_argument("--mask_img_dir",default="/scratch/connectome/3DCNN/data/1.ABCD/tmp/MNI152_T1_1mm_Brain_mask.nii.gz" ,required=False,help='')
parser.add_argument("--save_dir",default="/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing_BMI/XAI/result",required=False,help='')
args = parser.parse_args()




if __name__ == '__main__': 
    ## Gathering directories of IG map
    OBESITY_attr_dir = [] 
    OBESITY_attr_dir.append(os.path.join(*["/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing_BMI/XAI/result/attribute/become_overweight", args.study_sample+'_nifti', 'partition0']))
    OBESITY_attr_dir.append(os.path.join(*["/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing_BMI/XAI/result/attribute/become_overweight", args.study_sample+'_nifti', 'partition1']))
    OBESITY_attr_dir.append(os.path.join(*["/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing_BMI/XAI/result/attribute/become_overweight", args.study_sample+'_nifti', 'partition2']))
    OBESITY_attr_dir.append(os.path.join(*["/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing_BMI/XAI/result/attribute/become_overweight", args.study_sample+'_nifti', 'partition3']))
    OBESITY_attr_dir.append(os.path.join(*["/scratch/connectome/dhkdgmlghks/3DCNN_test/3DCNN_hardparameter_sharing_BMI/XAI/result/attribute/become_overweight", args.study_sample+'_nifti', 'partition4']))


    ## get every partition data
    OBESITY_file_list = []
    subject_list = [] 
    for OBESITY_attr_dir_partition in OBESITY_attr_dir:
        for file in  glob.glob(OBESITY_attr_dir_partition + '/*'): 
            OBESITY_file_list.append(file)
            subject_list.append(os.path.split(file)[-1].replace('.nii.gz', ''))
    OBESITY_subject = pd.DataFrame({'subjectkey': subject_list, 'file':OBESITY_file_list})


    ## loading phenotype data 
    phenotype = pd.read_csv('/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc/BMI_prediction/ABCD Release4.0 Tabular dataset.csv')

    ## loading mask image
    mask_img = nib.load(args.mask_img_dir)
    mask_affine = mask_img.affine 
    mask_header = mask_img.header
    mask_img = np.array(mask_img.dataobj)
    transform = Compose([AddChannel(), Resize((128, 128, 128))])
    mask_img = transform(mask_img)[0, :, :, :]
    mask_img = nib.Nifti1Image(mask_img, affine=mask_affine, header=mask_header)


    ## matching image file and phenotype file
    ##TODO## 
    """Integrating covariates for arguments"""
    """
    Using covariates only to fit model. 
    """
    
    #var_list = ['subjectkey'] + [target] + confounders
    num_covariate_list = ['age', 'income', 'high_educ'] 
    cat_covariate_list = ['sex', 'abcd_site', 'race_g']
    covariate_list = num_covariate_list + cat_covariate_list
    # remove duplicated variable
    if args.target in covariate_list: 
        raise ValueError("In this code, demgraphic variables could not be a target")
    var_list = ['subjectkey'] + [args.target] + covariate_list
    VBM_list = phenotype[phenotype['eventname'] == args.time_point][var_list]
    VBM_list['subjectkey'] = [subj.replace('_', '') for subj in VBM_list['subjectkey'].values]
    # remove subjects having NaN values in basic covariates 
    VBM_list = pd.merge(VBM_list, OBESITY_subject, how='inner', on='subjectkey')
    VBM_list = VBM_list.dropna(axis=0)
   
    # change categorical covariates into dummy variables 
    for cat in cat_covariate_list: 
        if VBM_list[cat].dtypes == 'str': 
            cat_dummies = pd.get_dummies(VBM_list[cat])
            VBM_list.drop([cat], axis=1, inplace=True)
            VBM_list = pd.concat([VBM_list, cat_dummies], axis=1)
        else: 
            cat_dummies = pd.get_dummies(VBM_list[cat].astype('str'))
            cat_dummies.columns = [cat + k for k in cat_dummies.keys()]
            VBM_list.drop([cat], axis=1, inplace=True)
            VBM_list = pd.concat([VBM_list, cat_dummies], axis=1)
    
        
    ## setting design matrix 
    #design_matrix = VBM_list[[target] + confounders]
    design_matrix = VBM_list.drop(['subjectkey', 'file'], axis=1)
    design_matrix['intercept'] = np.ones(len(design_matrix))
    design_matrix = design_matrix.astype('float')
    print("{} subjects are involved in this analyses".format(len(design_matrix)))
    print(design_matrix[args.target].value_counts())


    if args.permutation_test or args.tfce: 

        """
        ref: https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_second_level_association_test.html
        """
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there
        # is less than 10% probability to make a single false discovery
        # (90% chance that we make no false discoveries at all).
        # This threshold is much more conservative than the previous one.
        neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(second_level_input=list(VBM_list['file'].values),
                                                                       design_matrix=design_matrix,
                                                                       second_level_contrast=args.target,
                                                                       model_intercept=True, 
                                                                       n_perm=args.n_perm, 
                                                                       two_sided_test=False,
                                                                       mask=mask_img, 
                                                                       smoothing_fwhm=args.smoothing_fwhm, 
                                                                       tfce=args.tfce,
                                                                       verbose=1,
                                                                       n_jobs=args.n_process
                                                                       )
        neg_log_pvals_permuted_ols_filtered = np.where(neg_log_pvals_permuted_ols_unmasked.dataobj > -np.log10(args.n_perm),neg_log_pvals_permuted_ols_unmasked.dataobj, 0)
        neg_log_pvals_permuted_ols_filtered = nib.Nifti1Image(neg_log_pvals_permuted_ols_filtered, affine=neg_log_pvals_permuted_ols_unmasked.affine, header=neg_log_pvals_permuted_ols_unmasked.header)

        nib.save(neg_log_pvals_permuted_ols_filtered, os.path.join(args.save_dir, args.target+'_pval_map_Smoothing{}_threshold{}.nii.gz'.format(args.smoothing_fwhm, -np.log10(args.n_perm))))
        

    else: 
        """
        ref: https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_oasis.html
        """
        print("you should check that you properly set ALPHA and CORRECTION METHOD")
        ## fitting model
        second_level_model = SecondLevelModel(smoothing_fwhm=args.smoothing_fwhm, mask_img=mask_img)
        second_level_model.fit(list(VBM_list['file'].values), design_matrix=design_matrix)      # Using covariates only to fit model. 

        ## compute contrast map 
        contrast_map = second_level_model.compute_contrast(second_level_contrast=args.target, output_type='all')

        ## filtering contrast value of voxels with p value
        #thresholded_nifti, threshold = threshold_stats_img(z_map['p_value'], alpha=0.05, height_control='fdr')
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        p_val_img = np.where(contrast_map['p_value'].dataobj < (args.alpha / n_voxels), contrast_map['p_value'].dataobj, 1)
        thresholded_nifti = nib.Nifti1Image(p_val_img, affine=mask_affine, header=mask_header)

        nib.save(thresholded_nifti, os.path.join(args.save_dir, args.target+'_pval_map_Smoothing{}_Alpha{}.nii.gz'.format(args.smoothing_fwhm, args.alpha) ))
