'''
trains model to predict AD onset from EHR data
'''

import numpy as np
import sparse 
import matplotlib.pyplot as plt
import prepare_data_final
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import sklearn.calibration as calibration
from sklearn.utils import resample
import sklearn
import copy
import sys


'''
train model
'''
def train(training_in, training_lab, h, val_data=None):
    weight = 40
    mod = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=h, max_iter=250, tol=0.001, early_stopping=True, validation_fraction=0.1, n_iter_no_change=1, class_weight={0:1, 1:weight}) 
   mod.fit(training_in, training_lab)
    
    print(predict(mod, training_in, training_lab))
    
    return mod


'''
calibrate model
'''
def calibrate(mod, data, labels):
    mod_output = mod.decision_function(data).astype(float)
    prob_min = np.min(mod_output)
    prob_max = np.max(mod_output)
    mod_output = (mod_output - np.min(mod_output)) / (np.max(mod_output) - np.min(mod_output))
    num_bins = 5
    output_bins = np.array([1, 2, 3, 4, 5]) * ((max(mod_output) - min(mod_output))/5)

    bin_pred_avg = np.zeros((num_bins,))
    bin_actual_avg = np.zeros((num_bins,))
    for i in range(num_bins):
        if i == 0:
            bin_ind = np.where(mod_output <= output_bins[i])[0]
        else:
            bin_ind = np.where(np.logical_and(mod_output > output_bins[i - 1], mod_output <= output_bins[i]))[0]
        bin_pred_avg[i] = np.average(mod_output[bin_ind]) 
        bin_actual_avg[i] = np.average(labels[bin_ind]) 

    calibrating_mod = linear_model.LinearRegression(fit_intercept=False)
    cal_inp = np.concatenate(((bin_pred_avg ** 3).reshape(-1, 1), (bin_pred_avg ** 2).reshape(-1, 1), bin_pred_avg.reshape(-1, 1)), axis=1)
    calibrating_mod.fit(cal_inp, bin_actual_avg)

    return calibrating_mod, prob_min, prob_max


'''
test calibration
'''
def test_calibration(calibrating_mod, prob_min, prob_max, model, data, labels):
    mod_output = model.decision_function(data).astype(float)
    probs = (mod_output - prob_min) / (prob_max - prob_min)
    probs[probs > 1] = 1
    probs[probs < 0] = 0
    num_bins = 5
    bin_pred_avg = np.zeros((num_bins,))
    bin_actual_avg =  np.zeros((num_bins,))
    output_bins = np.array([1, 2, 3, 4, 5]) * ((max(probs) - min(probs))/5)
    for i in range(num_bins):
        if i == 0:
            bin_ind = np.where(probs <= output_bins[i])[0]
        else:
            bin_ind = np.where(np.logical_and(probs > output_bins[i - 1], probs <= output_bins[i]))[0]
        bin_pred_avg[i] = np.average(probs[bin_ind]) 
        bin_actual_avg[i] = np.average(labels[bin_ind])
    cal_inp = np.concatenate(((bin_pred_avg ** 3).reshape(-1, 1), (bin_pred_avg ** 2).reshape(-1, 1), bin_pred_avg.reshape(-1, 1)), axis=1)
    calibrated_probs = calibrating_mod.predict(cal_inp)
    '''plt.plot(calibrated_probs, bin_actual_avg, 'bo', label='Calibrated predictions')
    plt.plot([0, 0.05, 0.1], [0, 0.05, 0.1], 'orange', label='Perfect calibration')
    plt.title('Calibration Curve')
    plt.xlabel('Adjusted Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.show()'''

    return calibrated_probs, bin_actual_avg


'''
cross validation
'''
def cross_val(train_in, train_lab, hyperparams):
    num_folds = 5
    h = hyperparams[0] #reg const
    aucs = np.zeros((num_folds,))
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True)
    model = 0

    count = 0
    for train_ind, val_ind in folds.split(train_in, train_lab):
        train_fold_data, val_fold_data = train_in[train_ind], train_in[val_ind]
        train_fold_lab, val_fold_lab = train_lab[train_ind], train_lab[val_ind]
        val_data = [(val_fold_data, val_fold_lab)]
        model = train(train_fold_data, train_fold_lab, h, val_data)
        auc, conf_matr = (predict(model, val_fold_data, val_fold_lab, show_roc=False))
        aucs[count] = auc
        count += 1
        print(('val', auc, conf_matr))
    print(aucs)
    print(np.average(aucs))
    return np.average(aucs)


'''
predict classes of model output
'''
def class_prediction(model, inputs, thresh_percentile=65):
    mod_output = model.decision_function(inputs).astype(float)
    thresh = np.percentile(mod_output, thresh_percentile)

    pred = mod_output
    thresh_ind = np.where(mod_output >= thresh)[0]
    not_thresh_ind = np.where(mod_output < thresh)[0]
    pred[thresh_ind] = 1

    pred[not_thresh_ind] = 0
    pred = pred.astype(int)

    return pred


'''
predict on new data, return auc and confusion matrix
'''
def predict(model, inputs, labels, show_roc=False, thresh=65):
    mod_output = model.decision_function(inputs).astype(float) 
    auc = roc_auc_score(labels, mod_output)

    pred = class_prediction(model, inputs, thresh)
    conf_matr = sklearn.metrics.confusion_matrix(labels, pred)

    if show_roc:
        fp, tp, th = metrics.roc_curve(labels, mod_output)
        plt.plot(fp, tp)
        plt.plot([0, 0.5, 1], [0, 0.5, 1])
        plt.title('ROC Curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()
    
    return auc, conf_matr


'''
return coordinates for roc curve
'''
def get_roc(model, inputs, labels):
    mod_output = model.decision_function(inputs).astype(float) 
    auc = roc_auc_score(labels, mod_output)

    pred = class_prediction(model, inputs)
    conf_matr = sklearn.metrics.confusion_matrix(labels, pred)

    fp, tp, th = metrics.roc_curve(labels, mod_output)
    
    return fp, tp


'''
analysis of predictions - followup on converters based on time of conversion and prediction correctness
'''
def analyze_converters(model, inp, labs, pat_i):
    np.random.seed(0)
    num_bins = 6
    corr = np.zeros((0, num_bins))
    incorr = np.zeros((0, num_bins))
    corr_raw = np.zeros((0, num_bins))

    num_bootstrap = 1000
    num_inp = inp.shape[0]
    sample_prop = 1

    time_to_file = '/data2/Alzheimers/UMHS/68-72_new/all_converters_time_to.csv'
    pop_file = '/data2/Alzheimers/UMHS/68-72_new/pop.csv'
    time_to_raw = prepare_data_new.get_file(time_to_file, 2)
    test_pats = prepare_data_new.get_labels()[0][:, 0]
    test_pats = np.sort(test_pats)[pat_i]
    test_converters = np.isin(test_pats, time_to_raw[:, 0])
    all_conv_ind = np.where(test_converters)[0]
    conv_map = np.concatenate((all_conv_ind.reshape(1, -1), test_pats[all_conv_ind].reshape(1, -1)), axis=0)

    conv_bins = np.array([0., 30.6, 54.2, 77.8, 101.4, 120.5, 165])
    print(conv_bins)
    
    for j in range(num_bootstrap):
        inp_sample_i = np.random.choice(num_inp, int(num_inp * sample_prop), replace=True)
        labels = labs[inp_sample_i]
        inputs = inp[inp_sample_i, :]
        preds = class_prediction(model, inputs)

        converters = np.isin(inp_sample_i, all_conv_ind)
        num_converters = int(np.sum(converters))
        time_to = -1 * np.ones((num_converters, 2))
        
        conv_ind = inp_sample_i[np.where(converters)[0]] 
        conv_ind_boot = np.where(converters)[0]
        for i in range(num_converters):
            pat_ind = np.where(conv_map[0, :].astype(int) == conv_ind[i])[0]
            pat = conv_map[1, pat_ind][0]
            pat_time_to = time_to_raw[time_to_raw[:, 0] == pat, 1].astype(int)[0]
            if preds[conv_ind_boot[i]] == 0:
                time_to[i, 0] = pat_time_to 
            else:
                time_to[i, 1] = pat_time_to 
    
        time_to_wrong = time_to[:, 0][time_to[:, 0] >= 0]
        time_to_right = time_to[:, 1][time_to[:, 1] >= 0]
        n, b, p = plt.hist([time_to_wrong, time_to_right], bins=conv_bins, histtype='bar', stacked=True)
        denom = n[1]
        denom[denom == 0] = 1
        corr = np.append(corr, np.array([(n[1] - n[0]) / denom]), axis=0)
        incorr = np.append(incorr, np.array([n[0] / denom]), axis=0)
        corr_raw = np.append(corr_raw, np.array([n[1]]) / np.sum(np.array([n[1]])), axis=0)

    print('converter analysis')
    print(np.percentile(corr, [2.5, 50, 97.5], axis=0))
    print(np.average(corr, axis=0))
    print(np.percentile(incorr, [2.5, 50, 97.5], axis=0))
    print(np.percentile(corr_raw, [2.5, 50, 97.5], axis=0)) #corr raw is the total count for each time bin

    plt.clf()
    plt.bar([1, 2, 3, 4, 5, 6], np.percentile(corr, 50, axis=0), width=0.3, yerr=np.absolute(np.percentile(corr, [2.5, 97.5], axis=0) - np.percentile(corr, 50, axis=0)), label='Proportion\ncorrectly\npredicted\nto convert')
    plt.bar([1.3, 2.3, 3.3, 4.3, 5.3, 6.3], np.percentile(corr_raw, 50, axis=0), width=0.3, yerr=np.absolute(np.percentile(corr_raw, [2.5, 97.5], axis=0) - np.percentile(corr_raw, 50, axis=0)), label='Proprotion\namong\nconverters')
    plt.xlabel('Time to conversion (months)')
    plt.ylabel('Proportion')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.title('Classification among patients who convert to probable AD')
    plt.xticks([1, 2, 3, 4, 5, 6], ['0-30', '31-54', '55-77', '78-101', '102-120', '121-163'])
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


'''
find out who has memory issues
'''
def find_mem_prob(dementia=False):
    data_source = '0_align_250_t120.0003'
    raw_data = np.load('placeholder' + data_source + '/X.npz')
    data = sparse.COO(raw_data['coords'], raw_data['data'], tuple(raw_data['shape']))
    data = np.array(data.todense())

    feat_i = 2390
    if dementia:
        feat_i = 1704#1672 dementia 1704 parkinsons 1702 lewy body (no one has this)
    has_complaints = np.array([])
    for i in range(data.shape[1]):
        data_slice = data[:, i, :]
        has_complaints = np.union1d(has_complaints, np.where(data_slice[:, feat_i] != 0)[0])

    return np.unique(has_complaints)


'''
produces a model
hyperparams: reg const, num trees, tree depth, learn rate
'''
def get_model(data_source, use_cross_val, hyperparams):
    data = prepare_data_new.split_data(data_source, 'summary')
    train_data = data[0]
    train_labs = data[1]
    test_data = data[2]
    test_labs = data[3]
    test_pats = np.array(data[4]).astype(int)
    test_set = [(test_data, test_labs)]

    train_data2 = np.concatenate((train_data, test_data), axis=0)
    train_labs2 = np.concatenate((train_labs, test_labs))

    if use_cross_val:
        val_auc = cross_val(train_data, train_labs, hyperparams)
    model = train(train_data, train_labs, hyperparams[0], hyperparams[1], hyperparams[2], test_set)
    auc, conf_matr = predict(model, test_data, test_labs, show_roc=False)
    
    #uncomment the section below for calibration results
    '''num_folds = 2
    num_bootstrap = 1000
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True)
    for train_ind, val_ind in folds.split(test_data, test_labs):
        train_fold_data, val_fold_data = test_data[train_ind], test_data[val_ind]
        train_fold_lab, val_fold_lab = test_labs[train_ind], test_labs[val_ind]
        calibrating_mod, prob_min, prob_max = calibrate(model, train_fold_data, train_fold_lab)
        all_calibrated_probs = []
        all_bin_actual_avg = []
        brier_scores = []
        for i in range(num_bootstrap):
            boot_i = np.random.choice(val_fold_data.shape[0], val_fold_data.shape[0])
            val_boot_data = val_fold_data[boot_i, :]
            val_boot_lab = val_fold_lab[boot_i]
            c_probs, b_avg = test_calibration(calibrating_mod, prob_min, prob_max, model, val_boot_data, val_boot_lab)
            all_calibrated_probs.append(c_probs.reshape(1, -1))
            all_bin_actual_avg.append(b_avg.reshape(1, -1))
            brier_scores.append(np.sum(np.square(np.array(all_calibrated_probs) - np.array(all_bin_actual_avg))) / len(all_calibrated_probs))
        all_calibrated_probs = np.concatenate(all_calibrated_probs, axis=0)
        all_bin_actual_avg = np.concatenate(all_bin_actual_avg, axis=0)
        error_bars = np.zeros((2, all_calibrated_probs.shape[1]))
        error_bars[0, :] = np.percentile(all_calibrated_probs, 50, axis=0) - np.percentile(all_calibrated_probs, 2.5, axis=0)
        error_bars[1, :] = np.percentile(all_calibrated_probs, 97.5, axis=0) - np.percentile(all_calibrated_probs, 50, axis=0)
        plt.plot(np.median(all_bin_actual_avg, axis=0), all_calibrated_probs[0, :], 'b.', label='Calibrated predictions')
        plt.plot([0, 0.05, 0.3], [0, 0.05, 0.3], 'orange', label='Perfect calibration')
        plt.errorbar(np.median(all_bin_actual_avg, axis=0), all_calibrated_probs[0, :], yerr=error_bars, fmt='none')
        print(b_avg)
        print(all_calibrated_probs)
        print(np.percentile(brier_scores, [2.5, 50, 97.5]))
        plt.title('Calibration Curve')
        plt.ylabel('Adjusted Predicted Probability')
        plt.xlabel('Actual Probability')
        plt.legend(loc='upper left')
        plt.show()
        break'''

    if not use_cross_val:
        val_auc = auc
    print(auc)
    print(conf_matr)

    analyze_converters(model, test_data, test_labs, test_pats)
    
    return val_auc, model, test_data, test_labs


'''
bootstrapped results
'''
def get_bootstrapped_results(data_source, use_cross_val, hyperparams):
    data = prepare_data_new.split_data(data_source, 'summary')
    train_data = data[0]
    train_labs = data[1]
    test_data = data[2]
    test_labs = data[3]
    test_pats = np.array(data[4]).astype(int)
    test_set = [(test_data, test_labs)]

    model = train(train_data, train_labs, hyperparams[0], hyperparams[1], hyperparams[2], test_set)

    use_complaints = True
    if use_complaints:
        complaints = find_mem_prob(dementia=False)
        test_keep = np.where(np.isin(test_pats, complaints))[0]
        test_data = test_data[test_keep, :]
        test_labs = test_labs[test_keep]
        print('filtered test data: ', test_data.shape, test_labs.shape, np.sum(test_labs))

    num_bootstrap = 1000
    all_auc = []
    all_conf_matr = []
    all_tp = []
    fps = np.linspace(0, 1, 100)
    for i in range(num_bootstrap):
        boot_i = np.random.choice(test_data.shape[0], test_data.shape[0])
        test_boot_data = test_data[boot_i, :]
        test_boot_lab = test_labs[boot_i]
        if use_complaints:
            test_boot_data, test_boot_lab = resample(test_data, test_labs, stratify=test_labs)
        auc, conf_matr = predict(model, test_boot_data, test_boot_lab, show_roc=False)
        all_auc.append(auc)
        all_conf_matr.append(conf_matr)
        fp, tp = get_roc(model, test_boot_data, test_boot_lab)
        all_tp.append(np.interp(fps, fp, tp).reshape(1, -1))
    
    print(np.percentile(all_auc, [2.5, 50, 97.5]))
    print(np.percentile(all_conf_matr, [2.5, 50, 97.5], axis=0))

    tps = np.concatenate(all_tp, axis=0)
    plt.plot(fps, np.median(tps, axis=0), label='Median')
    plt.plot([0, 0.5, 1], [0, 0.5, 1], '--', color='k') 
    plt.fill_between(fps, np.percentile(tps, 2.5, axis=0), np.percentile(tps, 97.5, axis=0), alpha=0.2)
    plt.title('ROC Curve')
    plt.legend(loc='upper left')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()

    data_type = 'rocs/temp'
    np.save(data_type + '_fps.npy', fps)
    np.save(data_type + '_tps_median.npy', np.median(tps, axis=0))
    np.save(data_type + '_tps_bottom.npy', np.percentile(tps, 2.5, axis=0))
    np.save(data_type + '_tps_top.npy', np.percentile(tps, 97.5, axis=0))


'''
plot several roc curves
'''
def plot_many_rocs():
    colors = ['k', 'b', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    curves = ['All', 'Laboratory Tests', 'Procedures', 'Healthcare Utilization', 'Diagnoses', \
              'Vital Signs', 'Age Features', 'Medications', 'Demographics']
    aucs = ['AUROC: 0.70 (0.63-0.77)', 'AUROC: 0.63 (0.55-0.69)', 'AUROC: 0.60 (0.53-0.67)', \
            'AUROC: 0.56 (0.49-0.64)', 'AUROC: 0.55 (0.47-0.62)', 'AUROC: 0.54 (0.48-0.61)', \
            'AUROC: 0.54 (0.47-0.61)', 'AUROC: 0.51 (0.50-0.52)', 'AUROC: 0.50 (0.42-0.57)']
    curves = ['1000 Days Before', '500 Days Before', 'Alignment Only']
    aucs = ['AUROC: 0.70 (0.63-0.77)', 'AUROC: 0.63 (0.56-0.69)', 'AUROC: 0.54 (0.47-0.61)']
    plt.plot([0, 0.5, 1], [0, 0.5, 1], '--', color=colors[0])
   
    for i in range(len(curves)):
        fps = np.load('rocs/' + curves[i] + '_fps.npy')
        tps_med = np.load('rocs/' + curves[i] + '_tps_median.npy')
        tps_top = np.load('rocs/' + curves[i] + '_tps_top.npy')
        tps_bot = np.load('rocs/' + curves[i] + '_tps_bottom.npy') 
        plt.fill_between(fps, tps_bot, tps_top, alpha=0.2, color=colors[i + 1]) 
        if len(curves) == 3 and i < 2:
            plt.plot(fps, tps_med, label=curves[i][:-6] + 'Prior \n' + aucs[i], color=colors[i + 1])
        else:
            plt.plot(fps, tps_med, label=curves[i] + '\n' + aucs[i], color=colors[i + 1])
        
    plt.legend(loc='lower right', prop={'size': 8})
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
      

'''
get information on important weigths in linear model
'''
def analyze_weights(mod, num_feat_ret=3):
    sorted_features = np.argsort(mod.coef_)
    print('important weights')
    print(mod.coef_.shape)    

    if num_feat_ret != 'all':
       num_feat = num_feat_ret
       print(sorted_features[0, :num_feat])
       print(mod.coef_[0, sorted_features[0, :num_feat]])
       print(sorted_features[0, -num_feat:])
       print(mod.coef_[0, sorted_features[0, -num_feat:]])
       top = np.concatenate((sorted_features[0, :num_feat_ret], sorted_features[0, -num_feat_ret:]))
    else:
        top = sorted_features[0, :]
    return top        


'''
find feature meanings
'''
def weight_lookup(feat_i, feat_list=None):
    if feat_list is None:
        feat_list = 'placeholder/s.feature_names.txt'
    f1 = open(feat_list, 'r')
    c1 = f1.read()
    c1 = c1.split('\n')
    c1 = np.array(c1)
    c1 = c1[:-1]
    f1.close()
    features1 = c1

    feat_list2 = 'placeholder/X.feature_names.txt'
    f2 = open(feat_list2, 'r')
    c2 = f2.read()
    c2 = c2.split('\n')
    c2 = np.array(c2)
    c2 = c2[:-1]
    f2.close()
    features2 = c2

    if feat_i < features1.shape[0]:
        return features1[feat_i]

    feat_i = feat_i - features1.shape[0]
    feat_i_bin = int(feat_i / features2.shape[0])
    feat_i_lookup = feat_i % features2.shape[0]

    return str(feat_i_bin) + '-' + features2[feat_i_lookup]


'''
main block
'''
if __name__ == '__main__':
    np.random.seed(3)  

    hyperparam = [0.4] 
    get_bootstrapped_results('0_align_250_t120.0003', False, hyperparam)
    plot_many_rocs()

    '''
    hyperparams = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,1,10,100,1000]
    for h in hyperparams:
        get_bootstrapped_results('0_align_250_t120.0003', True, [h])
    '''

