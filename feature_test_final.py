import model_final as model1
import prepare_data_final
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import operator
import copy
import random


'''
find pairwise correlations
'''
def find_pair_corr(data):
    coefs = np.corrcoef(data, rowvar=False)
    coefs[np.isnan(coefs)] = 0 
    abs_coefs = np.absolute(coefs)
    coefs[abs_coefs < 0.1] = 0
    return coefs


'''
train model and find significant weights
'''
def find_important_weights(train_in, train_lab, test_in, test_lab):
    test_set = [(test_in, test_lab)]
    mod = model1.train(train_in, train_lab, 0.4, test_set)
    auc, conf_matr = model1.predict(mod, test_in, test_lab)
    print(auc)
    print(conf_matr)
    imp_weights = model1.analyze_weights(mod, 1000)
    print(imp_weights.shape)
    
    return mod, imp_weights


'''
aggregate groups of features
'''
def aggregate_feats(data):
    group_feats = {}
    all_feats = np.array([])
    for feat in data.keys():
        if not(feat in all_feats):
            group_feats[feat] = data[feat]
        else:
            for group in group_feats.keys():
                if feat in group_feats[group]:
                    group_feats[group] = np.concatenate((group_feats[group], data[feat]))
                    group_feats[group] = np.unique(group_feats[group])
        all_feats = np.concatenate((all_feats, data[feat]))
    return group_feats


'''
find correlated features of the given set
'''
def find_corr_feat(feats, data):
    correlation_thresh = 0.7
    coefs = find_pair_corr(data)
    feat_corrs = {}

    for feat in feats:
        corrs_sort_i = np.argsort(np.absolute(coefs[feat, :]))
        corrs_sort = np.absolute(coefs[feat, :])[corrs_sort_i]
        top_corrs = corrs_sort_i[corrs_sort > correlation_thresh]
        print(corrs_sort_i[corrs_sort > correlation_thresh])
        feat_corrs[feat] = np.flip(top_corrs)
        if feat_corrs[feat].shape[0] == 0:
            feat_corrs.pop(feat, None)

    return feat_corrs


'''
do permutation importance on model for the given features (and correlated features)
'''
def do_perm_imp(mod, feats, test, test_lab):
    np.random.seed(0)
    random.seed(0)
    init_auc, init_conf_matr = model1.predict(mod, test, test_lab)

    num_perm = 100
    only = np.zeros((num_perm,)) 
    all_feats = np.zeros((num_perm,)) 
    
    for i in range(num_perm):
        test_copy = copy.deepcopy(test)
        for j in range(feats.shape[0]):
            feat = feats[j] 
            perm = np.random.permutation(test_lab.shape[0])
            test_copy[:, feat] = test_copy[:, feat][perm]
            auc, conf_matr = model1.predict(mod, test_copy, test_lab)
            if j == 0:
                only[i] = 1 - auc
            if j == feats.shape[0] - 1:
                all_feats[i] = 1 - auc
    print((feats[0]))  
    return all_feats / (1-init_auc), \
        np.percentile(all_feats, 50) - (1 - init_auc), \
        np.percentile(all_feats, [5, 100]) - (1 - init_auc)


'''
permutation importance, except with bootstraps instead of repeated permutations
'''
def do_bootstrap_perm_imp(mod, feats, test, test_lab):
    init_auc, init_conf_matr = model1.predict(mod, test, test_lab)

    num_bootstrap = 100
    only = np.zeros((num_bootstrap,)) 
    all_feats = np.zeros((num_bootstrap,)) 
    bootstrap_sample_size = test.shape[0]

    test_copy = copy.deepcopy(test)
    for j in range(feats.shape[0]):
        feat = feats[j] 
        perm = np.random.permutation(test_lab.shape[0])
        test_copy[:, feat] = test_copy[:, feat][perm]
    
    for i in range(num_bootstrap):
        bootstrap_samples = np.random.choice(test.shape[0], bootstrap_sample_size).astype(int)
        test_lab_bootstrap = test_lab[bootstrap_samples]
        test_data_bootstrap = test_copy[bootstrap_samples, :]
        auc, conf_matr = model1.predict(mod, test_data_bootstrap, test_lab_bootstrap)
        only[i] = 1 - auc
        all_feats[i] = 1 - auc

    print((feats[0]))  
    plt.hist(all_feats - (1 - init_auc))
    plt.show()
    return all_feats / (1-init_auc), \
        np.percentile(all_feats, 50) - (1 - init_auc), \
        np.percentile(all_feats, [5, 100]) - (1 - init_auc)


'''
tests features identified as important by doing permutation importance
'''
def test_feat_importance(mod, imp_feats, test, test_lab, bootstrap=False):
    plots_data = []
    plots_data2 = []
    plot_labels = []
    plot_labels2 = []
    avg_diff = {}
    count = 0
    for feat in imp_feats.keys():
        print('feature number ' + str(count))
        coef = mod.coef_[0, feat]

        if bootstrap:
            data, diff, std = do_bootstrap_perm_imp(mod, imp_feats[feat], test, test_lab)
        else:
            data, diff, std = do_perm_imp(mod, imp_feats[feat], test, test_lab)
        avg_diff[(feat, model1.weight_lookup(feat), tuple(imp_feats[feat]), coef)] = (diff, tuple(std))

        if feat > 0 and feat < 9:
            plot_labels.append(feat)
            plots_data.append(data)
        else:
            plot_labels2.append(feat)
            plots_data2.append(data)
        count += 1

    avg_diff = sorted(avg_diff.items(), key=operator.itemgetter(1))
    num_sig_feats = 0
    for difference in avg_diff:
        if difference[1][1][0] > 0:
            print(difference)
            num_sig_feats += 1
    print(num_sig_feats)
    return 1


'''
main block
'''
if __name__ == '__main__':
    np.random.seed(3) 
    random.seed(0)

    data_source = '0_align_250_t120.0003'
    complete_data = prepare_data_new.split_data(data_source, 'summary')
    train_data = complete_data[0]
    train_labs = complete_data[1]
    test_data = complete_data[2]
    test_labs = complete_data[3]
    test_pats = complete_data[4]
    test_set = [(test_data, test_labs)]
    data = np.concatenate((train_data, test_data), axis=0)
    labels = np.concatenate((train_labs, test_labs))

    mod, imp_feats = find_important_weights(train_data, train_labs, test_data, test_labs)
    feat_correlations = find_corr_feat(imp_feats, data)

    feat_groups = aggregate_feats(feat_correlations)
    print(len(list(feat_groups.keys())))
    test_feat_importance(mod, feat_groups, test_data, test_labs, bootstrap=False)
    
