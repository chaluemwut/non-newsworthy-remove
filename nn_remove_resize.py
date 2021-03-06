from db.mysql_conn import MysqlDb
import pickle, random, copy
import numpy as np
from util import list_to_sql
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score
from tf_idf.document_similar import TFIDF
from tf_idf.pre_processing import PreProcessing
import time

db = MysqlDb()

def get_index(index_train, all_y):
    lst_1 = []
    lst_0 = []
    for index in index_train:
        data = all_y[index]
        if data == 1:
            lst_1.append(index)
        else:
            lst_0.append(index)
    
    return lst_1, lst_0
    
def tf_idf(tf_obj, msg, all_y, index_train, index_test):
    np_y = np.array(all_y)
    np_y_train = np_y[index_train]
    np_msg = np.array(msg)
    
    idx_lst_1, idx_lst_0 = get_index(index_train, all_y)
    
    tf_idf_training_size = int(len(idx_lst_1)*(.8))
    tf_idf_test_size = int(len(idx_lst_1)*(.2))
    
    idx_measure_corpus = idx_lst_1[0:tf_idf_training_size]
    data_test = idx_lst_1[tf_idf_training_size:(tf_idf_training_size+tf_idf_test_size)]
    data_test.extend(idx_lst_0[0:tf_idf_test_size])
    
    measure_corpus = np_msg[idx_measure_corpus]
    
    s_train_time = time.time()
    
    cosin_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    f1_lst = []
    for cosin in cosin_lst:
        y_true_lst = []
        y_pred_lst = []
        for data_test_idx in data_test:
            data = msg[data_test_idx]
            y_true = all_y[data_test_idx]
            y_true_lst.append(y_true)
            sim_max = tf_obj.max_sim_measure(measure_corpus, data)
            if sim_max > cosin:
                y_pred_lst.append(1)
            else:
                y_pred_lst.append(0)
        f1 = f1_score(y_true_lst, y_pred_lst)
        f1_lst.append(f1)
    f1_lst = np.array(f1_lst)
    f1_max_idx = f1_lst.argmax()
    cosin_max = cosin_lst[f1_max_idx]
    
    total_train_time = time.time() - s_train_time
    train_time = total_train_time/tf_idf_training_size
    
    s_predict_time = time.time()
    per_y_pred = []
    for idx_per_test in index_test:
        msg_per_test = msg[idx_per_test]
        sim_max = tf_obj.max_sim_measure(measure_corpus, msg_per_test)
        if sim_max > cosin_max:
            per_y_pred.append(1)
        else:
            per_y_pred.append(0)
    
    per_y_true = np_y[index_test]
    per_y_pred = np.array(per_y_pred)
    
    total_predict_time = time.time() - s_predict_time
    predict_time = total_predict_time/tf_idf_test_size
    
    return f1_score(y_true=per_y_true, y_pred=per_y_pred), train_time, predict_time

def ml(x_train, x_test, y_train, y_test):
    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    
    s_train_time = time.time()
    ml.fit(x_train, y_train)
    total_train = time.time() - s_train_time
    train_time = total_train/len(y_train)
    
    s_predict_time = time.time()
    y_pred = ml.predict(x_test)
    total_predict = time.time() - s_predict_time
    predict_time = total_predict/len(y_test)
    
    return f1_score(y_true=y_test, y_pred=y_pred), train_time, predict_time 

def get_all_data():
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    where_id = '({},{})'.format(','.join(measure), ','.join(un_measure))
    
    x = db.get_feature_by_row('('+','.join(measure)+')')
    y = [0]*len(x)
    
    x1 = db.get_feature_by_row('('+','.join(un_measure)+')')
    y1 = [1]*len(x1)
    
    x.extend(x1)
    y.extend(y1)
    msg_lst = db.get_text(where_id)
    return x, y, msg_lst

def process():
    all_x, all_y, msg_lst = get_all_data()
    pre_process = PreProcessing()
    all_doc = pre_process.process(msg_lst)
    tf_obj = TFIDF(all_doc)
    
    result_tf_idf = {}
    result_ml = {}
    result_time_tf_idf_train = {}
    result_time_tf_idf_predict = {}
    result_time_ml_train = {}
    result_time_ml_predict = {}
    test_size = ['0.2', '0.4', '0.6', '0.8']
    for s in test_size:
        per_tf_idf = []
        per_ml = []
        time_tf_idf_train_lst = []
        time_tf_idf_predict_lst = []
        
        time_ml_train_lst = []
        time_ml_predict_lst = []
        for i in range(0, 200):
            print ' training size ',s ,' i ', i
            r = random.randint(1, 10000)
            indx = range(0, len(all_y))
            x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(all_x, all_y, indx, test_size=float(s), random_state=r)
            
            f1_tf_idf, time_tf_df_train, time_tf_idf_predict = tf_idf(tf_obj, all_doc, all_y, indices_train, indices_test)
            f1_ml, time_ml_train, time_ml_predict = ml(x_train, x_test, y_train, y_test)
            
            per_tf_idf.append(f1_tf_idf)
            per_ml.append(f1_ml)
            
            time_tf_idf_train_lst.append(time_tf_df_train)
            time_tf_idf_predict_lst.append(time_tf_idf_predict)
            
            time_ml_train_lst.append(time_ml_train)
            time_ml_predict_lst.append(time_ml_predict)
            
        result_tf_idf[s] = per_tf_idf
        result_ml[s] = per_ml
        
        result_time_tf_idf_train[s] = time_tf_idf_train_lst
        result_time_tf_idf_predict[s] = time_tf_idf_predict_lst
        
        result_time_ml_train[s] = time_ml_train_lst
        result_time_ml_predict[s] = time_ml_predict_lst
        
    str_data_path = 'data/nn/'
    pickle.dump(result_time_tf_idf_train, open(str_data_path+'tf_idf_train_time.obj', 'wb'))
    pickle.dump(result_time_tf_idf_predict, open(str_data_path+'tf_idf_predict_time.obj', 'wb'))
    
    pickle.dump(result_time_ml_train, open(str_data_path+'supervise_train_time.obj', 'wb'))
    pickle.dump(result_time_ml_predict, open(str_data_path+'supervise_predict_time.obj', 'wb'))
        
    pickle.dump(result_tf_idf, open(str_data_path+'result_tf_id.obj','wb'))
    pickle.dump(result_ml, open(str_data_path+'result_ml.obj', 'wb'))

    for s in test_size:
        print 'test size ', s
        print 'f1 tf_idf ', np.average(result_tf_idf[s])
        print 'f1 ml ', np.average(result_ml[s])
        print 'time train tf_idf ', np.average(result_time_tf_idf_train[s])
        print 'time predict tf_id ', np.average(result_time_tf_idf_predict[s])
        print 'time train ml ', np.average(result_time_ml_train[s])
        print 'time predict ml ', np.average(result_time_ml_predict[s])    
    
if __name__ == '__main__':
    process()
