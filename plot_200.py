import pickle
# import matplotlib.pyplot as plt
import numpy as np

def plot_new():
    tf_id = pickle.load(open('data/nn/result_tf_id.obj', 'rb'))
    ml = pickle.load(open('data/nn/result_ml.obj', 'rb'))
    per_time = tf_id_train_teim = pickle.load(open('data/nn/tf_idf_test_time.obj', 'rb'))
    print np.average(per_time)
    for i in ['0.2', '0.4', '0.6', '0.8']:
        print 'i ',i
        per_tf_idf = tf_id[i]
        per_ml = ml[i]
        print np.average(per_tf_idf)
        print np.average(per_ml)
        
def plot_200():
    f1_ml = pickle.load(open('data/non_newsworthy_remove/supervise_predict_time','rb'))
    f1_tf_idf = pickle.load(open('data/non_newsworthy_remove/tf_idf_predict_time','rb'))
    number_of_predict = pickle.load(open('data/non_newsworthy_remove/number_test_data','rb'))
    for i in range(len(f1_ml)):
        perform_time = f1_tf_idf[i]
        num = number_of_predict[i]
        print perform_time/num
#     plt.plot(f1_tf_idf)
#     plt.plot(f1_ml)
#     plt.show()
    
if __name__ == '__main__':
    plot_new()