import pickle
import numpy as np
base_base = 'data/nn/data_mac'
test_size = ['0.8', '0.6', '0.4', '0.2']

def time_result():
    tf_idf_time = pickle.load(open(base_base + '/tf_idf_predict_time.obj', 'rb'))
    ml_time = pickle.load(open(base_base+'/supervise_predict_time.obj', 'rb'))
    for i in reversed(range(0, 4)):
        start = i*200
        end = start+200
        tf_id_obj_time = tf_idf_time[start:end]
        ml_obj_time = ml_time[start:end]
        print np.average(tf_id_obj_time) , ' ', np.average(ml_obj_time)

def performance_result():
    tf_idf = pickle.load(open(base_base + '/result_tf_id.obj', 'rb'))
    for i in test_size:
        print np.average(tf_idf[i])
    
if __name__ == '__main__':
    time_result()
