#verify cohort discovery tool
import numpy as np
import csv
import datetime
import operator
import pandas as pd
import copy
import matplotlib.pyplot as plt


'''
global variables
'''
file_root =  'placeholder'


'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[3:]
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    f.close()
    return c


'''
get the madc cohort (2 is female)
'''
def get_madc_cohort():
    files = []
    files.append(file_root + 'madc2.csv')
    files.append(file_root + 'madc3.csv')
    cohort = np.empty((0, 5))
    diagnosis_ind = 9
    first_row = True
    for file in files:
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            count = 1
            for row in reader:
                #print(row)
                if first_row:
                    first_row = False
                    continue
                new_row = np.array([[row[1], row[2], row[diagnosis_ind], row[3], row[4]]])
                cohort = np.append(cohort, new_row, axis=0)
                count += 1 
        diagnosis_ind = 8
        first_row = True 
    return cohort


'''
get mrns from the umhs cohort
'''
def get_umhs_cohort(data_type):
    file = file_root + 'placeholder' + data_type + '.csv'
    cohort = get_file(file, 3)
    return cohort


'''
get all overlapping patient visits
'''
def get_overlapping_visits():
    file = file_root + 'overlap_all_visits.csv'
    all_visits = get_file(file, 4)
    return all_visits


'''
get start and end of madrc time window
'''
def get_madrc_time_window(madrc_vis, window):
    vis_dates = []
    for k in range(madrc_vis.shape[0]):
        v = madrc_vis[k]
        vis_dates.append(np.datetime64(datetime.datetime.strptime(v[1], '%m/%d/%Y')))
    start = np.min(vis_dates) - window
    end = np.max(vis_dates) + window
    return start, end, np.sort(vis_dates)


'''
get overlap between 2 time windows
'''
def get_overlap(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end2, end1)
    overlap = max(overlap_end - overlap_start + np.timedelta64(1, 'D'), np.timedelta64(0, 'ns'))
    return overlap


'''
evaluate an individual person
returns 
    whether they are tp/fp/tn/fn 
    if they madc normal
    time between first rdw/madrc diagnosis if tp
    confusion matrix at encounter level
'''
def evaluate_individual(madrc, rdw, diagnosed_rdw, window):
    #get madrc time window
    madrc_start, madrc_end, madrc_dates = get_madrc_time_window(madrc, window)
    
    #find rdw time window
    rdw_start = np.min(rdw[:, 2].astype('datetime64')) 
    rdw_end = np.max(rdw[:, 2].astype('datetime64')) 
    
    #find madrc diagnosis
    ad_labels = np.where(np.core.defchararray.find(madrc[:, 2], 'AD') != -1)[0]
    poss_ad_labels = np.where(np.core.defchararray.find(madrc[:, 2], 'Possible') != -1)[0]
    ad_labels = np.setdiff1d(ad_labels, poss_ad_labels)
    madrc_ad = 0
    madrc_normal = 0
    first_ad = None
    if ad_labels.shape[0] > 0:
        madrc_ad = 1
        first_ad = madrc_dates[np.min(ad_labels)] - window
    elif np.unique(madrc[:, 2]).shape[0] == 1 and np.unique(madrc[:, 2])[0] == 'NL':
        madrc_normal = 1
    
    #find first rdw diagnosed visits if any
    first_rdw_diag = None
    if diagnosed_rdw is not None:
        first_rdw_diag = diagnosed_rdw[0, 1].astype('datetime64')
        first_rdw_diag -= window
        if first_rdw_diag > madrc_end:
            first_rdw_diag = None
    
    #find what should be returned (4 things above)
    time_overlap = get_overlap(madrc_start, madrc_end, rdw_start, rdw_end)
    time_overlap /= np.timedelta64(1, 'D')
    eval_decision = 0 #0 means unknown
    time_between_ad = 0
    time_to_fp = 0
    fn_to_ad = 0
    if madrc_ad and first_rdw_diag is not None:
        eval_decision = 'tp'
        time_between_ad = (first_ad - first_rdw_diag) / np.timedelta64(1, 'D')
    elif madrc_ad and time_overlap > 0:
        eval_decision = 'fn'
        fn_to_ad = (rdw_end - first_ad) / np.timedelta64(1, 'D')
    elif not madrc_ad and first_rdw_diag is not None and first_rdw_diag <= madrc_end:
        eval_decision = 'fp'
        time_to_fp = (madrc_end - first_rdw_diag) / np.timedelta64(1, 'D')
    elif (not madrc_ad and first_rdw_diag is None and time_overlap > 0) \
        or (not madrc_ad and first_rdw_diag is not None and first_rdw_diag > madrc_end):
        eval_decision = 'tn'
    
    possible_fns = 0
    if madrc_ad and first_rdw_diag is None and rdw_start > madrc_end:
        possible_fns += 1
        
    vis_conf_matr = np.zeros((2, 2))
    for i in range(rdw.shape[0]):
        visit_day = rdw[i, 2].astype('datetime64')
        ad_visit = False
        if first_rdw_diag is not None and visit_day >= first_rdw_diag:
            ad_visit = True
        if madrc_ad and ad_visit and visit_day >= first_ad:
            vis_conf_matr[1, 1] += 1 #tp
        elif (madrc_ad and ad_visit and visit_day < first_ad) \
             or (not madrc_ad and ad_visit):
            vis_conf_matr[0, 1] += 1 #fp
        elif (not madrc_ad and not ad_visit):
            vis_conf_matr[0, 0] += 1 #tn
        elif (madrc_ad and not ad_visit and visit_day >= first_ad):
            vis_conf_matr[1, 0] += 1 #fn
    
    return eval_decision, madrc_normal, time_between_ad, vis_conf_matr, \
           time_to_fp, fn_to_ad, possible_fns


'''
finds the population proportion adjusted ppv
'''
def find_adjusted_ppv(diagnosis, n, tp, fn, fp, fpn, num_n):
    adjust_const_n = 5
    adjust_const_d = 0.67
    if diagnosis == 'AD':
        adjust_const_n = 7.5
        adjust_const_d = 1.5
    denominator = tp \
                  + (fpn / num_n) * (tp + fn) * adjust_const_n \
                  + ((fp - fpn) / (n - num_n)) * (tp + fn) * adjust_const_d
    ppv_adjust = tp / denominator 
    return ppv_adjust


'''
plots for visualization
'''
def make_plots(tp, fp, fn):
    plt.hist(tp)
    plt.xlabel('MADRC Diagnosis - RDW Diagnosis')
    plt.ylabel('Count')
    plt.title('True Positives')
    plt.show()
    
    plt.hist(fp)
    plt.xlabel('End of MADRC Window - First Diagnosis')
    plt.ylabel('Count')
    plt.title('False Positives')
    plt.show()
    
    plt.hist(fn)
    plt.xlabel('Last RDW Visit - MADRC Diagnsosis')
    plt.ylabel('Count')
    plt.title('False Negatives')
    plt.show()
    
    return 1


'''
evaluate the cohort (based on madrc/rdw overalp)
'''
def evaluate_cohort(cohort_filter, data=None, ignore_pats=None):
    #get required data if needed
    if data is None:
        madrc_vis = get_madc_cohort()
        cohort = get_umhs_cohort(cohort_filter)
        rdw_vis = get_overlapping_visits()
        print(cohort.shape[0])
    else:
        madrc_vis = data[0]
        cohort = data[1]
        rdw_vis = data[2]
    
    if ignore_pats is not None:
        include_pats = np.where(1 - np.isin(rdw_vis[:, 1].astype(int), ignore_pats))[0]
        rdw_vis = rdw_vis[include_pats, :]
        
    #setup
    window = np.timedelta64(185, 'D')
    eval_pats = np.unique(rdw_vis[:, 1].astype(int))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    #extra things to keep track of
    fpn = 0
    num_normal = 0 #normal by madc diagnoses
    vis_conf_matr = np.array([[0, 0], [0, 0]])
    avg_time_between, tb = 0, []
    avg_time_to_fp, tfp = 0, []
    avg_fn_to_ad, tad = 0, []
    possible_fns = 0
    
    #loop through everyone that needs to be evaluated
    for i in range(eval_pats.shape[0]):
        pat = eval_pats[i]
        num_occur = np.where(cohort[:, 0].astype(int) == pat)[0].shape[0]
        if num_occur == 0:
            num_occur = 1
        pat_madrc_vis = madrc_vis[madrc_vis[:, 0].astype(int) == pat, :]
        if max(pat_madrc_vis[:, 3].astype(int)) < 65:
            continue
        pat_rdw_vis = rdw_vis[rdw_vis[:, 1].astype(int) == pat, :]
        pat_diagnosed_vis = None
        if pat in cohort[:, 0].astype(int):
            pat_diagnosed_vis = cohort[cohort[:, 0].astype(int) == pat, :]
        eval_outcome = evaluate_individual(pat_madrc_vis, pat_rdw_vis, pat_diagnosed_vis, window)
            
        if eval_outcome[0] == 'tp':
            tp += 1 * num_occur
            avg_time_between += eval_outcome[2] * num_occur
            tb.append(eval_outcome[2] * num_occur)
        elif eval_outcome[0] == 'fp':
            fp += 1 * num_occur
            avg_time_to_fp += eval_outcome[4] * num_occur
            tfp.append(eval_outcome[4] * num_occur)
            if eval_outcome[1] == 1:
                fpn += 1 * num_occur
        elif eval_outcome[0] == 'tn':
            tn += 1 * num_occur
        elif eval_outcome[0] == 'fn':
            fn += 1 * num_occur
            avg_fn_to_ad += eval_outcome[5] * num_occur
            tad.append(eval_outcome[5] * num_occur)
        
        if eval_outcome[1] == 1 and eval_outcome[0] != 0:
            num_normal += 1 * num_occur
        
        possible_fns += eval_outcome[6] * num_occur
        
        if eval_outcome[0] != 0:
            vis_conf_matr += eval_outcome[3].astype(int) * num_occur
    
    #calculate metrics
    sens = tp / (tp + fn)
    spef = tn / (tn + fp)
    ppv = tp / (tp + fp)
    ppv_adj = find_adjusted_ppv('AD', tn + fp, tp, fn, fp, fpn, num_normal)
    avg_time_between /= tp
    avg_time_to_fp /= fp
    avg_fn_to_ad /= fn
    
    #plot distribution of various times to ad/false positive ad
    #make_plots(tb, tfp, tad)
    #print(tp, fp, tn, fn, fpn, num_normal)
    #print(np.max(tb))
    #print(possible_fns)
    
    return sens, spef, ppv, ppv_adj, avg_time_between, vis_conf_matr, \
           avg_time_to_fp, avg_fn_to_ad
           

'''
bootstrap
'''
def bootstrap(cohort_filter, num_bootstraps=1000):
    madc = get_madc_cohort()
    umhs = get_umhs_cohort(cohort_filter)
    rdw_vis = get_overlapping_visits()
        
    sen = np.zeros((num_bootstraps,))
    spef = np.zeros((num_bootstraps,))
    ppv =  np.zeros((num_bootstraps,))
    ppv_adj = np.zeros((num_bootstraps,))
    time_between = np.zeros((num_bootstraps,))
    enc_conf_matr = np.zeros((num_bootstraps, 2, 2))
    to_fp = np.zeros((num_bootstraps,))
    to_fn = np.zeros((num_bootstraps,))

    prop_sample = 1
    bootstrap_size_umhs = int(umhs.shape[0] * prop_sample)

    for i in range(num_bootstraps):
        umhs_sample = umhs[np.random.choice(umhs.shape[0], bootstrap_size_umhs)]
        extra_pats = np.setdiff1d(np.intersect1d(umhs[:, 0].astype(int), madc[:, 0].astype(int)), umhs_sample[:, 0].astype(int))
        sample_sen, sample_spef, sample_ppv, sample_ppv_a, sample_tb, sample_enc_conf_matr, \
            sample_fp, sample_fn = evaluate_cohort(cohort_filter, [madc, umhs_sample, rdw_vis], extra_pats)
        sen[i] = sample_sen
        spef[i] = sample_spef
        ppv[i] = sample_ppv
        ppv_adj[i] = sample_ppv_a
        time_between[i] = sample_tb
        enc_conf_matr[i, :, :] = sample_enc_conf_matr
        to_fp[i] = sample_fp
        to_fn[i] = sample_fn
    
    print('sen', np.average(sen), np.percentile(sen, [2.5, 50, 97.5]))
    print('spef', np.average(spef), np.percentile(spef, [2.5, 50, 97.5]))
    print('ppv', np.average(ppv), np.percentile(ppv, [2.5, 50, 97.5]))
    print('ppv adjust', np.average(ppv_adj), np.percentile(ppv_adj, [2.5, 50, 97.5]))
    print('time_between', np.average(time_between), np.percentile(time_between, [2.5, 50, 97.5]))
    print('enc_conf_matr', np.average(enc_conf_matr, axis=0), np.percentile(enc_conf_matr, [2.5, 50, 97.5], axis=0))
    print('avg time to fp', np.average(to_fp), np.percentile(to_fp, [2.5, 50, 97.5]))
    print('avg fn to ad', np.average(to_fn), np.percentile(to_fn, [2.5, 50, 97.5]))

    return np.percentile(sen, [2.5, 50, 97.5]), np.percentile(spef, [2.5, 50, 97.5]), np.percentile(ppv, [2.5, 50, 97.5]), np.percentile(time_between, [2.5, 50, 97.5])


'''
main block
'''
if __name__ == '__main__':
    np.random.seed(0)

    diag_types = ['AD'] 
    for diag_type in diag_types:
        print(diag_type)
        cohort_filters = ['icdad', 'published']
        '''cohort_filters = ['icd' + diag_type.lower(), 'rx', 'psych', 'neuro', \
                    'icd' + diag_type.lower() + 'ANDrx', 'icd' + diag_type.lower() + 'ANDpsych', 'icd' + diag_type.lower() + 'ANDneuro']'''
        '''cohort_filters = ['icd' + diag_type.lower(), 'rx', 'psych', 'neuro', \
                    'icd' + diag_type.lower() + 'ANDrx', 'icd' + diag_type.lower() + 'ANDpsych', 'icd' + diag_type.lower() + 'ANDneuro', \
                    'rxANDpsych', 'rxANDneuro', 'psychANDneuro', \
                    'icd' + diag_type.lower() + 'ANDrxANDpsych', 'icd' + diag_type.lower() + 'ANDrxANDpsychANDneuro', \
                    'icd' + diag_type.lower() + 'ORrxORpsych', 'icd' + diag_type.lower() + 'ORrxORpsychORneuro']'''
        for cohort_filter in cohort_filters:
            print(cohort_filter)
            #print('population')
            evaluation_results = evaluate_cohort(cohort_filter)
            print(evaluation_results)
            print('boostrap')
            #cohort = get_umhs_cohort(cohort_filter)
            #bootstrap(cohort_filter, num_bootstraps=1000) #start 5:12pm
            print('\n')

'''
sen 0.6992231281526967 [0.64516129 0.70168364 0.74418605]
spef 0.91664203982581 [0.88974191 0.91666667 0.94214876]
ppv 0.7452653091965721 [0.66666667 0.74553429 0.81818182]
ppv adjust 0.7668395027909012 [0.71009093 0.76567678 0.82303485]
time_between -9.054581226230347 [-117.48623596   -7.09270121   92.51769949]
enc_conf_matr [[29246.834  4064.618]
 [ 1235.926  5886.979]] [[[27027.825  2324.175]
  [  901.9    4107.9  ]]

 [[29185.5    4064.   ]
  [ 1224.     5890.   ]]

 [[31757.25   6076.4  ]
  [ 1641.125  7825.025]]]
avg time to fp 542.8422636504788 [404.0525584  541.60421836 691.30886364]
avg fn to ad 2344.9915416349722 [2233.706875   2344.56907895 2461.02278409]
'''
