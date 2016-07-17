from db.mysql_conn import MysqlDb
from sklearn import tree
from sklearn.metrics import f1_score
import numpy as np

import pickle, random, copy
db = MysqlDb()

def str_join(lst):
    return '(' + ','.join(lst) + ')'

def process1(x_test, y_test):    
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    
    random.shuffle(measure)
    random.shuffle(measure)
    
    random.shuffle(un_measure)
    random.shuffle(un_measure) 
    
    measure = measure[0:100]
    un_measure = un_measure[0:100]
    
    adding_measure = measure[0:len(measure)]
    adding_unmeasure = un_measure[0:10]
        
    x_train = []
    y_train = []
    
    x_measure = db.get_feature_by_row(str_join(adding_measure))
    y_measure = db.get_label_data(str_join(adding_measure))
    x_train.extend(x_measure)
    y_train.extend(y_measure)
        
    x_un_measure = db.get_feature_by_row(str_join(adding_unmeasure))
    y_un_measure = db.get_label_data(str_join(adding_unmeasure))
    x_train.extend(x_un_measure)
    y_train.extend(y_un_measure)
    
    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    return f1_score(y_test, y_pred)

def process2(x_test, y_test):    
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    
    random.shuffle(measure)
    random.shuffle(measure)
    
    random.shuffle(un_measure)
    random.shuffle(un_measure) 
    
    measure = measure[0:100]
    un_measure = un_measure[0:100]
    
    n = 0
    adding_measure = measure[0:len(measure)]
    adding_unmeasure = un_measure[0:n]
        
    x_train = []
    y_train = []
    
    x_measure = db.get_feature_by_row(str_join(adding_measure))
    y_measure = db.get_label_data(str_join(adding_measure))
    x_train.extend(x_measure)
    y_train.extend(y_measure)

    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    return f1_score(y_test, y_pred)

def process_un_measure(n, x_test, y_test):    
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    
    random.shuffle(measure)
    random.shuffle(measure)
    
    random.shuffle(un_measure)
    random.shuffle(un_measure) 
    
    measure = measure[0:100]
    un_measure = un_measure[0:100]
    
    n_measure = len(measure) - n
    adding_measure = measure[0:n_measure]
    adding_unmeasure = un_measure[0:n]
        
    x_train = []
    y_train = []
    
    if n != 100:
        x_measure = db.get_feature_by_row(str_join(adding_measure))
        y_measure = db.get_label_data(str_join(adding_measure))
        x_train.extend(x_measure)
        y_train.extend(y_measure)
        
    if n != 0:
        x_un_measure = db.get_feature_by_row(str_join(adding_unmeasure))
        y_un_measure = db.get_label_data(str_join(adding_unmeasure))
        x_train.extend(x_un_measure)
        y_train.extend(y_un_measure)
    
    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    return f1_score(y_test, y_pred)

def get_random_training_data():
    num_row = 1300
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    str_lst = '({},{})'.format(','.join(measure), ','.join(un_measure))
     
    random_id_list = [str(x) for x in db.get_random_training_data(str_lst, num_row)];
    str_id = '(' + ','.join(random_id_list) + ')'
    x_test = db.get_feature_by_row(str_id)
    y_test = db.get_label_data(str_id) 
    return x_test, y_test   

def drop_perf():
    print 'test'
    fsc1 = []
    fsc2 = []
    for i in range(0, 200):
        x_test, y_test = get_random_training_data()                
        fsc1.append(process1(x_test, y_test))
        fsc2.append(process2(x_test, y_test))

#     pickle.dump(fsc1, open('data/socinfo/fsc1.obj', 'wb'))
#     pickle.dump(fsc2, open('data/socinfo/fsc2.obj', 'wb'))        
    print np.average(fsc1)
    print np.average(fsc2)
        
if __name__ == '__main__':
    drop_perf()