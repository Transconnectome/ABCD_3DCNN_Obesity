import os 
import copy
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import pearsonr, ttest_ind 
from scipy.cluster.hierarchy import dendrogram, linkage 
from scipy.cluster.hierarchy import fcluster 

from statsmodels.stats.multitest import multipletests
import multiprocessing
from tqdm import tqdm 
from functools import partial


import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import warnings 
warnings.filterwarnings("ignore")



def clustering_KMEANS(base_dir, save_dir, cluster_save_dir, year, data, n_clusters, scaler, attr_score):
    if data.find('HarvardOxford-cort'):
        save_dir = os.path.join(save_dir, 'HarvardOxford_Cortical')
    elif data.find('HarvardOxford-sub'):
        save_dir = os.path.join(save_dir, 'HarvardOxford_Subcortical')
    elif data.find('Cerebellum'):
        save_dir = os.path.join(save_dir, 'Cerebellum')
    elif data.find('CIT168toMNI152-FSL'):
        save_dir = os.path.join(save_dir, 'ReinforcementLearning')
    elif data.find('CIT168_iAmyNuc'):
        save_dir = os.path.join(save_dir, 'Amygdala')

    # filtering case subject 
    if year == 'after1y':
        phenotype_dir  = '/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/dnn/ABCD_phenotype_total_1years_become_overweight_10PS_stratified_partitioned_5fold.csv'
    elif year == 'after2y': 
        phenotype_dir  = '/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/dnn/ABCD_phenotype_total_2years_become_overweight_10PS_stratified_partitioned_5fold.csv'
    pheno = pd.read_csv(phenotype_dir)
    case_subject_list = list(pheno[pheno['become_overweight'] == 1]['subjectkey'].values)
    case_subject_list = pd.DataFrame({'subjectkey': case_subject_list})
    control_subject_list = list(pheno[pheno['become_overweight'] == 0]['subjectkey'].values)
    control_subject_list = pd.DataFrame({'subjectkey': control_subject_list})
    attr_score_case = pd.merge(attr_score, case_subject_list, how='inner', on='subjectkey')
    # fitering case subject for delta 
    pheno_delta = pd.read_csv('/Users/wangheehwan/Desktop/CNN_for_BMI/phenotype_data/ABCD Release4.0 Tabular dataset.csv')
    for i in range(len(pheno_delta)): 
        pheno_delta['subjectkey'][i] = pheno_delta['subjectkey'][i].replace('_','')
    pheno_delta_filtered = pd.merge(pheno_delta, case_subject_list, how='inner', on='subjectkey')

    # filtering out subjectkey and zero values
    for k in attr_score_case.keys(): 
        if k == 'subjectkey':
            attr_score_case = attr_score_case.drop('subjectkey', axis=1, inplace=False)
        else: 
            if np.sum(attr_score_case[k].values) == 0: 
                attr_score_case = attr_score_case.drop(k, axis=1, inplace=False)
    X = attr_score_case.values 
    scaled_X = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_X)
    print(silhouette_score(scaled_X, cluster_labels))
    cluster_label_pheno = pd.merge(attr_score, case_subject_list, how='inner', on='subjectkey')
    cluster_label_pheno['cluster_label'] = cluster_labels
    pheno_cluster = pd.merge(pheno, cluster_label_pheno[['subjectkey', 'cluster_label']], how='inner', on='subjectkey')
    pheno_cluster_delta = pd.merge(pheno_delta_filtered, cluster_label_pheno[['subjectkey', 'cluster_label']], how='inner', on='subjectkey')
    pheno_control = pd.merge(pheno, control_subject_list, how='inner', on='subjectkey')
    pheno_delta_control = pd.merge(pheno_delta, control_subject_list, how='inner', on='subjectkey')

    if data.find('HarvardOxford-cort') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'HarvardOxford_Cortical')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('HarvardOxford-sub') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'HarvardOxford_Subcortical')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('Cerebellum') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'Cerebellum')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('CIT168toMNI152-FSL') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'ReinforcementLearning')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('CIT168_iAmyNuc') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'Amygdala')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)

    return pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control




def clustering_HIERARCHICAL(base_dir, save_dir, cluster_save_dir, year, data, cutoff, scaler, attr_score):
    if data.find('HarvardOxford-cort'):
        save_dir = os.path.join(save_dir, 'HarvardOxford_Cortical')
    elif data.find('HarvardOxford-sub'):
        save_dir = os.path.join(save_dir, 'HarvardOxford_Subcortical')
    elif data.find('Cerebellum'):
        save_dir = os.path.join(save_dir, 'Cerebellum')
    elif data.find('CIT168toMNI152-FSL'):
        save_dir = os.path.join(save_dir, 'ReinforcementLearning')
    elif data.find('CIT168_iAmyNuc'):
        save_dir = os.path.join(save_dir, 'Amygdala')

    # filtering case subject 
    if year == 'after1y':
        phenotype_dir  = '/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/dnn/ABCD_phenotype_total_1years_become_overweight_10PS_stratified_partitioned_5fold.csv'
    elif year == 'after2y': 
        phenotype_dir  = '/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/dnn/ABCD_phenotype_total_2years_become_overweight_10PS_stratified_partitioned_5fold.csv'
    pheno = pd.read_csv(phenotype_dir)
    case_subject_list = list(pheno[pheno['become_overweight'] == 1]['subjectkey'].values)
    case_subject_list = pd.DataFrame({'subjectkey': case_subject_list})
    control_subject_list = list(pheno[pheno['become_overweight'] == 0]['subjectkey'].values)
    control_subject_list = pd.DataFrame({'subjectkey': control_subject_list})
    attr_score_case = pd.merge(attr_score, case_subject_list, how='inner', on='subjectkey')
    # fitering case subject for delta 
    pheno_delta = pd.read_csv('/Users/wangheehwan/Desktop/CNN_for_BMI/phenotype_data/ABCD Release4.0 Tabular dataset.csv')
    for i in range(len(pheno_delta)): 
        pheno_delta['subjectkey'][i] = pheno_delta['subjectkey'][i].replace('_','')
    pheno_delta_filtered = pd.merge(pheno_delta, case_subject_list, how='inner', on='subjectkey')

    # filtering out subjectkey and zero values
    for k in attr_score_case.keys(): 
        if k == 'subjectkey':
            attr_score_case = attr_score_case.drop('subjectkey', axis=1, inplace=False)
        else: 
            if np.sum(attr_score_case[k].values) == 0: 
                attr_score_case = attr_score_case.drop(k, axis=1, inplace=False)
    X = attr_score_case.values 
    scaled_X = scaler.fit_transform(X)

    linked = linkage(scaled_X, 'ward')

    print(silhouette_score(scaled_X, fcluster(linked, cutoff, criterion='distance')-1))
    cluster_label_pheno = pd.merge(attr_score, case_subject_list, how='inner', on='subjectkey')
    cluster_label_pheno['cluster_label'] = fcluster(linked, cutoff, criterion='distance')-1
    pheno_cluster = pd.merge(pheno, cluster_label_pheno[['subjectkey', 'cluster_label']], how='inner', on='subjectkey')
    pheno_cluster_delta = pd.merge(pheno_delta_filtered, cluster_label_pheno[['subjectkey', 'cluster_label']], how='inner', on='subjectkey')
    pheno_control = pd.merge(pheno, control_subject_list, how='inner', on='subjectkey')
    pheno_delta_control = pd.merge(pheno_delta, control_subject_list, how='inner', on='subjectkey')


    if data.find('HarvardOxford-cort') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'HarvardOxford_Cortical')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('HarvardOxford-sub') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'HarvardOxford_Subcortical')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('Cerebellum') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'Cerebellum')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('CIT168toMNI152-FSL') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'ReinforcementLearning')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)
    elif data.find('CIT168_iAmyNuc') != -1:
        cluster_save_dir = os.path.join(cluster_save_dir, 'Amygdala')
        pheno_cluster.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster.csv'.format(year)), index=False)
        pheno_cluster_delta.to_csv(os.path.join(cluster_save_dir, '{}_case_cluster_delta.csv'.format(year)), index=False)
        pheno_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster.csv'.format(year)), index=False)
        pheno_delta_control.to_csv(os.path.join(cluster_save_dir, '{}_control_cluster_delta.csv'.format(year)), index=False)

    return pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control




def performing_ttest(title, target_list, year, pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control, save_dir, delta=True):
    if year == 'after1y': 
        followup_year = "1_year_follow_up_y_arm_1"
    elif year =='after2y':
        followup_year = '2_year_follow_up_y_arm_1' # choices = ["1_year_follow_up_y_arm_1", "2_year_follow_up_y_arm_1"] 
    num_processes = multiprocessing.cpu_count()

    ### raw score
    ## summary template
    df_final = []
    with multiprocessing.Pool(processes=num_processes) as pool: 
        with tqdm(total=len(target_list)) as pbar: 
            for target, result_tmp in tqdm(pool.imap_unordered(partial(ttest_multiprocess, pheno_cluster=pheno_cluster,pheno_control=pheno_control), target_list)):
                df_target = pd.DataFrame({'test': ['cluster0-cluster1'],
                                'variables': [target],
                                'T-statistic': [result_tmp.statistic],
                                'p_value': [result_tmp.pvalue]})
                df_final.append(df_target)
                pbar.update()
    df_final = pd.concat(df_final)
    ## p value correction 
    _, pval_adj, _, _ = multipletests(df_final['p_value'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
    df_final['p_value_BONF'] = pval_adj
    df_final.to_csv(os.path.join(save_dir, "{}_{}_ttest.csv".format(title, year)), index=False)

    if delta:
        ### delta score
        ## summary template
        df_delta_final = [] 
        with multiprocessing.Pool(processes=num_processes) as pool: 
            with tqdm(total=len(target_list)) as pbar: 
                for target, result_tmp in tqdm(pool.imap_unordered(partial(ttest_multiprocess_delta, followup_year=followup_year, pheno_cluster_delta=pheno_cluster_delta, pheno_delta_control=pheno_delta_control), target_list)):
                    df_delta_target = pd.DataFrame({'test': ['cluster0-cluster1'],
                                    'variables': [target],
                                    'T-statistic': [result_tmp.statistic],
                                    'p_value': [result_tmp.pvalue]})                    
                    df_delta_final.append(df_delta_target)
                    pbar.update()
        df_delta_final = pd.concat(df_delta_final)
        ## pvalue correction 
        _, delta_pval_adj, _, _ = multipletests(df_delta_final['p_value'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
        df_delta_final['p_value_BONF'] = delta_pval_adj
        df_delta_final.to_csv(os.path.join(save_dir, "{}_{}_delta_ttest.csv".format(title, year)), index=False)        




def ttest_multiprocess(target, pheno_cluster,pheno_control):
    ## data preprocess
    # control 
    df_control = pheno_control[[target]] 
    df_control.dropna(axis=0, inplace=True)
    # case
    df = pheno_cluster[['cluster_label',  target]] 
    df.dropna(axis=0, inplace=True)

    ## T-test with raw score
    # cluster0-cluster1
    result_cluster0_cluster1 = ttest_ind(df[(df['cluster_label'] == 0)][target].values, df[(df['cluster_label'] == 1)][target].values, permutations=10000000)
    return target, result_cluster0_cluster1



def ttest_multiprocess_delta(target, followup_year, pheno_cluster_delta, pheno_delta_control):
    ## data preprocess
    # control delta 
    df_delta_control = pheno_delta_control[['subjectkey', 'eventname', target]] 
    df_delta_control.dropna(axis=0, inplace=True)
    baseline_control = df_delta_control[df_delta_control['eventname'] == 'baseline_year_1_arm_1'][['subjectkey',  target]] 
    baseline_control.columns = ['subjectkey', '%s_baseline' % target]     # cluster_label_baseline = cluster_label_follow
    followup_control = df_delta_control[df_delta_control['eventname'] == followup_year][['subjectkey', target]] 
    followup_control.columns = ['subjectkey', '%s_followup' % target]
    df_delta_control = pd.merge(baseline_control, followup_control, how='inner', on='subjectkey')
    df_delta_control['delta'] = df_delta_control['%s_baseline' % target].values - df_delta_control['%s_followup' % target].values
    # case delta
    df_delta_case = pheno_cluster_delta[['subjectkey','cluster_label', 'eventname', target]] 
    df_delta_case.dropna(axis=0, inplace=True)
    baseline_case = df_delta_case[df_delta_case['eventname'] == 'baseline_year_1_arm_1'][['subjectkey','cluster_label',  target]] 
    baseline_case.columns = ['subjectkey', 'cluster_label_baseline', '%s_baseline' % target]     # cluster_label_baseline = cluster_label_follow
    followup_case = df_delta_case[df_delta_case['eventname'] == followup_year][['subjectkey','cluster_label',  target]] 
    followup_case.columns = ['subjectkey', 'cluster_label_followup', '%s_followup' % target]
    df_delta_case = pd.merge(baseline_case, followup_case, how='inner', on='subjectkey')
    df_delta_case['delta'] = df_delta_case['%s_baseline' % target].values - df_delta_case['%s_followup' % target].values

    ## T-test with delta score(after 1y - baseline score)
    # cluster0-cluster1
    result_delta_cluster0_cluster1 = ttest_ind(df_delta_case[(df_delta_case['cluster_label_baseline'] == 0)]['delta'].values, df_delta_case[(df_delta_case['cluster_label_baseline'] == 1)]['delta'].values, permutations=1000000)
    return target, result_delta_cluster0_cluster1




if __name__ == '__main__': 
    ### global setting
    base_dir =  "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/attr_score"
    save_dir = "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/ttest"
    cluster_save_dir = "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/cluster_label"
    target_title_dict = {'Demographic': ['BMI_sds_baseline', 'birth_weight', 'child_age', 'high_educ', 'rh_income1'],  
                        'PolygenicScore': [ 'bmieur4', 'edauto', 'insomniaeur6', 'snoringeur1', 'adhdeur6', 'ptsdeur4', 'bipauto',  'mddeur6',  'anxietyauto', 'ocdauto', 'worryauto'] , 
                        'SleepDisturbance': ['sleep_disturb_dims', 'sleep_disturb_sbd', 'sleep_disturb_da', 'sleep_disturb_swtd', 'sleep_disturb_does', 'sleep_disturb_shy', 'sleep_disturb_total'], 
                        'CBCLSyndrome': ["cbcl_anxiety", "cbcl_withdep","cbcl_somatic","cbcl_social","cbcl_thought","cbcl_attention","cbcl_rulebreak","cbcl_aggressive","cbcl_internal","cbcl_external","cbcl_totprob"], 
                        'CBCLDSM': ["cbcl_dsm_depression","cbcl_dsm_anxiety","cbcl_dsm_somatic","cbcl_dsm_adhd","cbcl_dsm_opposit","cbcl_dsm_conduct"], 
                        'CBCL2007': ["cbcl_sct"	,"cbcl_ocd","cbcl_stress"], 
                        'Screentime_main': ["screentime_wkday_y", "screentime_wkend_y", "screentime_wkday_p", "screentime_wkend_p"], 
                        'Screentime_sub': ["screentime_wkday_tv","screentime_wkday_videos",	"screentime_wkday_games","screentime_wkday_texting","screentime_wkday_sns", "screentime_wkday_videochat", "screentime_wkend_tv", "screentime_wkend_videos", "screentime_wkend_games", "screentime_wkend_texting", "screentime_wkend_sns", "screentime_wkend_videochat", "screentime_maturegames", "screentime_rmovies"]}
    

    ### after1y using K-means clustering
    year = "after1y"
    title_list = ['Demographic',  'PolygenicScore', 'SleepDisturbance', 'CBCLSyndrome', 'CBCLDSM', 'CBCL2007', 'Screentime_main', 'Screentime_sub']
    Kmeans_list = ["HarvardOxford_Subcortical", "Amygdala", "Cerebellum"]
    for roi in Kmeans_list: 
        # setting hyper parameters
        if roi == 'HarvardOxford_Cortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-cort-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'HarvardOxford_Subcortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-sub-maxprob-thr25-1mm_MeanAttrScore_revised.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'ReinforcementLearning':
            data = os.path.join(*[base_dir, year, "CIT168toMNI152-FSL_det_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == "Amygdala": 
            data = os.path.join(*[base_dir, year, "CIT168_iAmyNuc_1mm_MNI_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi ==  "Cerebellum": 
            data = os.path.join(*[base_dir, year, "Cerebellum-MNIfnirt-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        n_clusters = 2 
        attr_score = pd.read_csv(data)
        scaler = StandardScaler() 
        pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control = clustering_KMEANS(base_dir, save_dir_roi, cluster_save_dir, year, data, n_clusters, scaler, attr_score=attr_score)
        # performing t test 
        for title in title_list: 
            target_list = target_title_dict[title]
            for target in target_list: 
                if (not target in pheno_cluster.keys()) or (not target in pheno_cluster_delta.keys()) or (not target in pheno_control.keys()) or (not target in pheno_delta_control.keys()):
                    target_list.remove(target)
            if (title == 'Demographic')  or (title == 'PolygenicScore'):
                if title == title == 'PolygenicScore': 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster[pheno_cluster['euro'] == 1], pheno_cluster_delta=pheno_cluster_delta[pheno_cluster_delta['euro']==1], pheno_control=pheno_control[pheno_control['euro']==1], pheno_delta_control=pheno_delta_control[pheno_delta_control['euro'] ==1], save_dir=save_dir_roi, delta=False)
                else: 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=False)
            else:
                performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=True)


    ### after1y Heirarchical clustering 
    year = "after1y"
    title_list = ['Demographic',  'PolygenicScore', 'SleepDisturbance', 'CBCLSyndrome', 'CBCLDSM', 'CBCL2007', 'Screentime_main', 'Screentime_sub']
    heirarch_list = ["HarvardOxford_Cortical","ReinforcementLearning"]
    cutoff_list = [70, 50]
    for roi, cutoff in zip(heirarch_list, cutoff_list): 
        # setting hyper parameters
        if roi == 'HarvardOxford_Cortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-cort-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'HarvardOxford_Subcortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-sub-maxprob-thr25-1mm_MeanAttrScore_revised.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'ReinforcementLearning':
            data = os.path.join(*[base_dir, year, "CIT168toMNI152-FSL_det_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == "Amygdala": 
            data = os.path.join(*[base_dir, year, "CIT168_iAmyNuc_1mm_MNI_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi ==  "Cerebellum": 
            data = os.path.join(*[base_dir, year, "Cerebellum-MNIfnirt-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        attr_score = pd.read_csv(data)
        scaler = StandardScaler() 
        pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control = clustering_HIERARCHICAL(base_dir, save_dir, cluster_save_dir, year, data, cutoff, scaler, attr_score)
        # performing t test 
        for title in title_list: 
            target_list = target_title_dict[title]
            for target in target_list: 
                if (not target in pheno_cluster.keys()) or (not target in pheno_cluster_delta.keys()) or (not target in pheno_control.keys()) or (not target in pheno_delta_control.keys()):
                    target_list.remove(target)
            if (title == 'Demographic')  or (title == 'PolygenicScore'):
                if title == title == 'PolygenicScore': 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster[pheno_cluster['euro'] == 1], pheno_cluster_delta=pheno_cluster_delta[pheno_cluster_delta['euro']==1], pheno_control=pheno_control[pheno_control['euro']==1], pheno_delta_control=pheno_delta_control[pheno_delta_control['euro'] ==1], save_dir=save_dir_roi, delta=False)
                else: 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=False)
            else:
                performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=True)

    
    ### after2y K-means clustering
    year = "after2y"
    title_list = ['Demographic',  'PolygenicScore', 'SleepDisturbance', 'CBCLSyndrome', 'CBCLDSM', 'CBCL2007']
    Kmeans_list = ["Amygdala", "Cerebellum"]
    for roi in Kmeans_list: 
        # setting hyper parameters
        if roi == 'HarvardOxford_Cortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-cort-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'HarvardOxford_Subcortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-sub-maxprob-thr25-1mm_MeanAttrScore_revised.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'ReinforcementLearning':
            data = os.path.join(*[base_dir, year, "CIT168toMNI152-FSL_det_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == "Amygdala": 
            data = os.path.join(*[base_dir, year, "CIT168_iAmyNuc_1mm_MNI_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi ==  "Cerebellum": 
            data = os.path.join(*[base_dir, year, "Cerebellum-MNIfnirt-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        n_clusters = 2 
        attr_score = pd.read_csv(data)
        scaler = StandardScaler() 
        pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control = clustering_KMEANS(base_dir, save_dir, cluster_save_dir, year, data, n_clusters, scaler, attr_score)
        # performing t test 
        for title in title_list: 
            target_list = target_title_dict[title]
            for target in target_list: 
                if (not target in pheno_cluster.keys()) or (not target in pheno_cluster_delta.keys()) or (not target in pheno_control.keys()) or (not target in pheno_delta_control.keys()):
                    target_list.remove(target)
            if (title == 'Demographic')  or (title == 'PolygenicScore'):
                if title == title == 'PolygenicScore': 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster[pheno_cluster['euro'] == 1], pheno_cluster_delta=pheno_cluster_delta[pheno_cluster_delta['euro']==1], pheno_control=pheno_control[pheno_control['euro']==1], pheno_delta_control=pheno_delta_control[pheno_delta_control['euro'] ==1], save_dir=save_dir_roi, delta=False)
                else: 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=False)
            else:
                performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=True)

    ### after2y Heirarchical clustering
    year = "after2y"
    title_list = ['Demographic',  'PolygenicScore', 'SleepDisturbance', 'CBCLSyndrome', 'CBCLDSM', 'CBCL2007']
    heirarch_list = ["HarvardOxford_Cortical", "HarvardOxford_Subcortical", "ReinforcementLearning"]
    cutoff_list = [70, 60, 50]
    for roi, cutoff in zip(heirarch_list, cutoff_list): 
        # setting hyper parameters
        if roi == 'HarvardOxford_Cortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-cort-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'HarvardOxford_Subcortical': 
            data = os.path.join(*[base_dir, year, "HarvardOxford-sub-maxprob-thr25-1mm_MeanAttrScore_revised.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == 'ReinforcementLearning':
            data = os.path.join(*[base_dir, year, "CIT168toMNI152-FSL_det_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi == "Amygdala": 
            data = os.path.join(*[base_dir, year, "CIT168_iAmyNuc_1mm_MNI_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        elif roi ==  "Cerebellum": 
            data = os.path.join(*[base_dir, year, "Cerebellum-MNIfnirt-maxprob-thr25-1mm_MeanAttrScore.csv"])
            save_dir_roi = os.path.join(*[save_dir, roi])
        attr_score = pd.read_csv(data)
        scaler = StandardScaler() 
        pheno_cluster, pheno_cluster_delta, pheno_control, pheno_delta_control = clustering_HIERARCHICAL(base_dir, save_dir, cluster_save_dir, year, data, cutoff, scaler, attr_score)
        # performing t test 
        for title in title_list: 
            target_list = target_title_dict[title]
            for target in target_list: 
                if (not target in pheno_cluster.keys()) or (not target in pheno_cluster_delta.keys()) or (not target in pheno_control.keys()) or (not target in pheno_delta_control.keys()):
                    target_list.remove(target)
            if (title == 'Demographic')  or (title == 'PolygenicScore'):
                if title == title == 'PolygenicScore': 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster[pheno_cluster['euro'] == 1], pheno_cluster_delta=pheno_cluster_delta[pheno_cluster_delta['euro']==1], pheno_control=pheno_control[pheno_control['euro']==1], pheno_delta_control=pheno_delta_control[pheno_delta_control['euro'] ==1], save_dir=save_dir_roi, delta=False)
                else: 
                    performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=False)
            else:
                performing_ttest(title=title, target_list=target_list, year=year, pheno_cluster=pheno_cluster, pheno_cluster_delta=pheno_cluster_delta, pheno_control=pheno_control, pheno_delta_control=pheno_delta_control, save_dir=save_dir_roi, delta=True)