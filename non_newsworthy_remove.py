from db.mysql_conn import MysqlDb
import pickle, random, copy
import numpy as np
from util import list_to_sql
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score
from tf_idf.document_similar import TFIDF
from tf_idf.pre_processing import PreProcessing

db = MysqlDb()

def ml(x_train, x_test, y_train, y_test):
    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    return f1_score(y_test, y_pred)  

def get_all_data():
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    where_id = '({},{})'.format(','.join(measure), ','.join(un_measure))
    
    x = db.get_feature_by_row('('+','.join(measure)+')')
    y = [1]*len(x)
    
    x1 = db.get_feature_by_row('('+','.join(un_measure)+')')
    y1 = [0]*len(x1)
    
    x.extend(x1)
    y.extend(y1)
    msg_lst = db.get_text(where_id)
    return x, y, msg_lst

def process():
    f1_tf_idf_lst = []
    f1_ml_lst = []
    all_x, all_y, msg_lst = get_all_data()
    pre_process = PreProcessing()
    all_doc = pre_process.process(msg_lst)
    tf_obj = TFIDF(msg_lst)
    
    for _ in range(0, 200):
        r = random.randint(1, 10000)
        indx = range(0, len(all_y))
        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(all_x, all_y, indx, test_size=0.2, random_state=r)
        f1_tf_idf = 10
        f1_ml = ml(x_train, x_test, y_train, y_test)
        f1_tf_idf_lst.append(f1_tf_idf)
        f1_ml_lst.append(f1_ml)
    
#     pickle.dump(f1_tf_idf_lst, open('data/non_newsworthy_remove/f1_tf_idf.obj', 'wb'))
#     pickle.dump(f1_ml_lst, open('data/non_newsworthy_remove/f1_ml.obj', 'wb'))
    
    print 'tf idf ', np.average(f1_tf_idf_lst)
    print 'ml ', np.average(f1_ml_lst)
    
if __name__ == '__main__':
        process()