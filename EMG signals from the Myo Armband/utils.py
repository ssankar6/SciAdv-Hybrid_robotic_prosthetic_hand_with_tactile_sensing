import scipy.io as sio
import numpy as np

def loadmat(filename):
#    '''
#    this function should be called instead of direct sio.loadmat
#    as it cures the problem of not properly recovering python dictionaries
#    from mat files. It calls the function check keys to cure all entries
#    which are still mat-objects
#    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = elem#_tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

## N Class Data Set Flattener 
#ab_5class_train_flat = []  #[NUM_SUJECTS, NUM_TIMESTEPS, #CHANNELS]
#ab_5class_train_label = [] 

def flatten_set(sub_data, SUBJECT_LIST, CLASS_LIST, data_range, num_channels = 128):
    samples_flat = []  #[NUM_SUJECTS, NUM_TIMESTEPS, #CHANNELS]
    labels_flat = []
    
    T = len(np.array(sub_data[SUBJECT_LIST[0]][1]['RE'])[0])
    num_trials = len(data_range)
    
    for subject in SUBJECT_LIST:
        i = 0
        samples = np.zeros((num_channels, num_trials*len(CLASS_LIST)*T))
        labels = []
        for trial in data_range:
            for c in CLASS_LIST:
                samples[:,i:i+T] = sub_data[subject][trial][c] 
                new_labels = [0]*len(CLASS_LIST)
                new_labels[CLASS_LIST.index(c)] = 1
                new_labels = [new_labels]*(T)
                labels  = [*labels, *new_labels] 
                i = i+T
        samples_flat.append(np.transpose(samples))
        labels_flat.append(labels)
    
    return (samples_flat, labels_flat)

## Data Set Shaper 

def  shape_set(X, Y, num_trials, S_LEN):      
    #for i in range(len(ab_5class_train_flat)):
    
        ########################################### left off here 
    
    Y_ = np.array(Y)
    num_classes = len(Y_[0])
    NUM_CHAN = len(X[0])
    T = int(len(X[:, 0])/(num_classes*num_trials))
    class_splits = np.array([*range(num_trials*num_classes)])*T
        
    num_samples = num_trials*num_classes*(T-S_LEN)
    X_m = np.zeros((num_samples,S_LEN,NUM_CHAN))
    Y_m = np.zeros((num_samples,S_LEN,num_classes))
    
    j = 0
    for i in class_splits:
        samples = X[i:i+T,:]
        ysamples = Y_[i:i+T,:]
        
        
        for t in range(S_LEN+1,T):
            sample = samples[t-S_LEN:t,:]
            ysample = ysamples[t-S_LEN:t,:]


            
            X_m[j,:,:] = sample
            Y_m[j,:,:] = ysample
            j = j+1
    
    return (X_m, Y_m)
    
def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    #gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return total_memory
