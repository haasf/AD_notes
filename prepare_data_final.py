'''
prepares data 
'''

import numpy as np
import sparse 


'''
global variable, used when preprocessing
'''
deltat = 250
T = 3500


'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[1:]
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    return c


'''
normalize data
'''
def normalize(data):
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    
    dims = data.shape
    mins = np.tile(mins, (dims[0], 1))
    maxs = np.tile(maxs, (dims[0], 1))
    
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    data[:, :] = (data[:,:] - mins[:, :]) / (ranges)
    return data


'''
load time invariant data
'''
def load_time_invar(data_source):
    raw_data = np.load('placeholder' + data_source + '/s.npz')
    data = sparse.COO(raw_data['coords'], raw_data['data'], tuple(raw_data['shape']))
    data = data.todense()

    pat_trajectories = get_file('placeholder', 2)
    pats = pat_trajectories[:, 0]
    pat_sort_i = np.argsort(pats)

    data = data[pat_sort_i, :]

    return data


'''
flip time variant data
'''
def flip_data(data, traj_bins):
    count = 0
    for bin_num in traj_bins:
        data[count, :bin_num + 1, :] = np.flip(data[count, :bin_num + 1, :], axis=1)
        count += 1
    return data


'''
load time variant data
mod_type determines whether to use summary of sequence data
data_types can be summary of sequential
    summary: concatenates sum of first 2 elements of time sequence and last 8 (values clipped at 1)
    sequential: leaves data as is and returns a sequence of length 10
'''
def load_time_var(data_source, data_type):
    raw_data = np.load('placeholder' + data_source + '/X.npz')
    data = sparse.COO(raw_data['coords'], raw_data['data'], tuple(raw_data['shape']))
    data = np.array(data.todense())

    pat_trajectories = get_file('placeholder', 2)
    traj_lens = pat_trajectories[:, 1].astype(int)
    traj_lens[np.where(traj_lens > T)[0]] = T
    traj_bins = (traj_lens // deltat).astype(int) 
    pat_sort_i = np.argsort(pat_trajectories[:, 0])
    traj_bins = traj_bins[pat_sort_i]

    if data_type == 'summary':
        #process variable lengths patient trajectories    
        num_segs = 4 #number of segments per feature vector
        seg_size = 1 #number of bins per segment
        ret_data = np.zeros((data.shape[0], 0))
        for i in range(num_segs):
            segment = np.sum(data[:, i*seg_size:(i*seg_size) + seg_size, :], axis=1)
            ret_data = np.concatenate((ret_data, segment), axis=1)
    else:
        ret_data = flip_data(data, traj_bins) #prepare_sequences(data)
    
    ret_data = normalize(ret_data)
    return ret_data


'''
prepare sequences for input
'''
def prepare_sequences(data):
    traj_lens = get_file('placeholder', 2)
    traj_lens = traj_lens[:, 1].astype(int)
    traj_bins = (traj_lens / deltat).astype(int) 

    inputs = []
    
    for i in range(traj_bins.shape[0]):
        inputs.append(data[i, :traj_bins[i] + 1, :])

    sorted_order = np.flip(np.argsort(traj_bins))
    traj_bins = traj_bins[sorted_order]
    inputs = [inputs[i] for i in sorted_order]
 
    return inputs 


'''
combine time in/variant data
'''
def combine_data(data_source, data_type, var_len=False):
    time_invar = load_time_invar(data_source)
    time_var = load_time_var(data_source, data_type)

    #if using sequence data, need to tile time_invar
    if len(time_var.shape) == 3:
        temp = np.zeros((time_invar.shape[0], time_var.shape[1], time_invar.shape[1]))
        #loop over sequence length
        for i in range(time_var.shape[1]): 
            temp[:, i, :] = time_invar
        time_invar = temp
        all_data = np.concatenate((time_invar, time_var), axis=2)
        if var_len:
            all_data = prepare_sequences(all_data)
    #using summary data, concatenate on features
    else:
        #all_data = time_var
        all_data = np.concatenate((time_invar, time_var), axis=1) #use everything
        '''dems = time_invar #demographics only
        icds = time_var[:, 1416:2720] #icds only
        cpts = time_var[:, 162:1395] #cpts only
        labs = time_var[:, 2721:3832] #labs only
        util = np.concatenate((time_var[:, 0].reshape(-1, 1), time_var[:, 3935:3939], time_var[:, 3948:3964]), axis=1) #util
        age = np.concatenate((time_var[:, 0].reshape(-1, 1), time_var[:, 3948:3964]), axis=1)
        vist = time_var[:, 3935:3939] #vistype
        vits = np.concatenate((time_var[:, 11:41], time_var[:, 122:127], \
                   time_var[:, 134:139], time_var[:, 1407:1412]), axis=1) #vitals
        meds = np.concatenate((time_var[:, 1:11], time_var[:, 41:122], \
                   time_var[:, 127:134], time_var[:, 139:162], time_var[:, 1395:1407], \
                   time_var[:, 1412:1416], time_var[:, 2720].reshape(-1, 1), \
                   time_var[:, 3832:3935], time_var[:, 3939:3948]), axis=1) #meds
        if time_var.shape[1] > 3964:
            num_segs = int(time_var.shape[1] / 3964)
            for i in range(1, num_segs):
                icds = np.concatenate((icds, time_var[:, (i*3964)+1416:(i*3964)+2720]), axis=1) 
                cpts = np.concatenate((cpts, time_var[:, (i*3964)+162:(i*3964)+1395]), axis=1) 
                labs = np.concatenate((labs, time_var[:, (i*3964)+2721:(i*3964)+3832]), axis=1) 
                util = np.concatenate((util, time_var[:, (i*3964)+0].reshape(-1, 1), \
                           time_var[:, (i*3964)+3935:(i*3964)+3939], time_var[:, (i*3964)+3948:(i*3964)+3964]), axis=1) 
                age = np.concatenate((util, time_var[:, (i*3964)+0].reshape(-1, 1), \
                           time_var[:, (i*3964)+3948:(i*3964)+3964]), axis=1) 
                vist = np.concatenate((vist, time_var[:, (i*3964)+3935:(i*3964)+3939]), axis=1) 
                vits = np.concatenate((vits, time_var[:, (i*3964)+11:(i*3964)+41], time_var[:, (i*3964)+122:(i*3964)+127], \
                           time_var[:, (i*3964)+134:(i*3964)+139], time_var[:, (i*3964)+1407:(i*3964)+1412]), axis=1) 
                meds = np.concatenate((meds, time_var[:, (i*3964)+1:(i*3964)+11], time_var[:, (i*3964)+41:(i*3964)+122], \
                           time_var[:, (i*3964)+127:(i*3964)+134], time_var[:, (i*3964)+139:(i*3964)+162], \
                           time_var[:, (i*3964)+1395:(i*3964)+1407], \
                           time_var[:, (i*3964)+1412:(i*3964)+1416], time_var[:, (i*3964)+2720].reshape(-1, 1), \
                           time_var[:, (i*3964)+3832:(i*3964)+3935], time_var[:, (i*3964)+3939:(i*3964)+3948]), axis=1) 
        #all_data = np.concatenate((cpts, labs, util, icds, vits, dems), axis=1)
        all_data = dems'''

    return all_data


'''
gets patient labels
first column is patient id
second columns is the time from alignment to conversion (if they convert, otherwise -1)
third column is label
'''
def get_labels():
    labels_file = 'placeholder' 
    labels = get_file(labels_file, 4)
    labels_sort_i = np.argsort(labels[:, 0])
    labels = labels[labels_sort_i, :]

    #remove extra patients
    pats = np.sort(get_file('placeholder', 1)[1:].reshape(-1))
    excluded_patients = np.where(np.logical_not(np.isin(labels[:, 0], pats)))[0]
    labels = np.delete(labels, excluded_patients, axis=0)
    print(labels.shape)

    excluded_patients2 = np.where(np.logical_not(np.isin(pats, labels[:, 0])))[0]
    print(excluded_patients2.shape)
    return labels, excluded_patients2


'''
split data into training and test set
in addition to training and test split, returns indexes of test patients
'''
def split_data(data_source, data_type):
    #setup
    data = combine_data(data_source, data_type)
    labels_raw, exclude_pats = get_labels()#[:, 2].astype(int)
    labels = np.logical_and(labels_raw[:, 2].astype(int) == 1, labels_raw[:, 1].astype(int) <= 120)
    data = np.delete(data, exclude_pats, axis=0)
    print(np.sum(labels), labels.shape, data.shape)
        
    prop_training = 0.8
    num_features = data.shape[1] if len(data.shape) != 3 else data.shape[2]
    pos_pats = np.where(labels == 1)[0]
    neg_pats = np.where(labels == 0)[0]
    training_labs = np.zeros((0,))
    test_labs = np.zeros((0,))
    test_pat_i = np.zeros((0,))

    if len(data.shape) != 3:
        training_data = np.zeros((0, num_features)) 
        test_data = np.zeros((0, num_features)) 
    else:
        training_data = np.zeros((0, data.shape[1], num_features))
        test_data = np.zeros((0, data.shape[1], num_features))
    
    #stratified split
    for pat_set in [pos_pats, neg_pats]:
        num_pats = pat_set.shape[0]
        rand_perm = np.random.permutation(num_pats)
        num_training = int(prop_training * num_pats)
        pat_labs = labels[pat_set][rand_perm]
        training_labs = np.append(training_labs, pat_labs[:num_training])
        test_labs = np.append(test_labs, pat_labs[num_training:])
        test_pat_i = np.append(test_pat_i, pat_set[rand_perm][num_training:])

        if len(data.shape) != 3:
            pat_data = data[pat_set, :][rand_perm, :]
            training_data = np.append(training_data, pat_data[:num_training, :], axis=0)
            test_data = np.append(test_data, pat_data[num_training:, :], axis=0)
        else:
            pat_data = data[pat_set, :, :][rand_perm, :, :]
            training_data = np.append(training_data, pat_data[:num_training, :, :], axis=0)
            test_data = np.append(test_data, pat_data[num_training:, :, :], axis=0)

    #randomize order of positive/negative examples
    train_perm = np.random.permutation(training_labs.shape[0])
    test_perm = np.random.permutation(test_labs.shape[0])

    training_labs = training_labs[train_perm]
    test_labs = test_labs[test_perm]
    test_pat_i = test_pat_i[test_perm]

    if len(data.shape) != 3:
        training_data = training_data[train_perm, :]
        test_data = test_data[test_perm, :]
    else:
        training_data = training_data[train_perm, :, :]
        test_data = test_data[test_perm, :, :]

    return (training_data, training_labs, test_data, test_labs, test_pat_i)


'''
splits squential_data
'''
def split_data_sequential(data_source):
    #setup
    data = combine_data(data_source, 'sequential', var_len=True)
    labels = get_labels()[:, 2].astype(int)

    prop_training = 0.8
    pos_pats = np.where(labels == 1)[0]
    neg_pats = np.where(labels == 0)[0]
    training_labs = np.zeros((0,))
    test_labs = np.zeros((0,))
    test_pat_i = np.zeros((0,))

    training_data = [] 
    test_data = [] 
    
    #stratified split
    for pat_set in [pos_pats, neg_pats]:
        num_pats = pat_set.shape[0]
        rand_perm = np.random.permutation(num_pats)
        num_training = int(prop_training * num_pats)
        pat_labs = labels[pat_set][rand_perm]
        training_labs = np.append(training_labs, pat_labs[:num_training])
        test_labs = np.append(test_labs, pat_labs[num_training:])
        test_pat_i = np.append(test_pat_i, pat_set[rand_perm][num_training:])

        pat_data = [data[i] for i in pat_set] 
        pat_data = [pat_data[i] for i in rand_perm]
        training_data = training_data + pat_data[:num_training] 
        test_data = test_data + pat_data[num_training:] 

    #randomize order of positive/negative examples
    train_perm = np.random.permutation(training_labs.shape[0])
    test_perm = np.random.permutation(test_labs.shape[0])

    training_labs = training_labs[train_perm]
    test_labs = test_labs[test_perm]

    training_data = [training_data[i] for i in train_perm]
    test_data = [test_data[i] for i in test_perm] 

    print('done splitting')
    return (training_data, training_labs, test_data, test_labs, test_pat_i)


'''
main block
'''
if __name__ == '__main__':
    pass
    #find_mem_prob()
    #data_type = 'summary'
    #split_data_sequential('first_align')
