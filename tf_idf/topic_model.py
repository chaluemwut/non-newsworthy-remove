# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
from pre_processing import Loader, PreProcessing
import pickle


def topic_model(documents, doc_test, fun_str, is_tf_idf):
    texts = documents
    dictionary =  corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    if is_tf_idf != True :
        tfidf = models.TfidfModel(corpus, id2word=dictionary)
        corpus_tfidf = tfidf[corpus]
    #     model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    #     model = models.RpModel(corpus_tfidf, num_topics=300)
        model = eval(fun_str)
    else:
        model = eval(fun_str)

    vec_bow = dictionary.doc2bow(doc_test)
    vec_space = model[vec_bow]
    index = similarities.MatrixSimilarity(model[corpus])
    sims = index[vec_space]
    
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims, dictionary

def get_key(search_value, dict):
    for key, value in dict.iteritems():
        if value == search_value :
            print key
            return key, value
 
def print_result(documents, doc_test, str_method, is_tf_idf):
    sim_obj = topic_model(documents, doc_test, str_method, is_tf_idf)[0]
    ind = sim_obj[0]
    print '--------------------------------'
    print ' '.join(doc_test), '|||', ind[1], ' ||| ', ' '.join(documents[ind[0]])
            
if __name__ == '__main__':
    documents = pickle.load(open('data/obj/filter.obj'))
    agree_filter = pickle.load(open('data/obj/agree_filter.obj'))
    str_lst_func = ['models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)', 
                    'models.RpModel(corpus_tfidf, num_topics=300)']
    str_lst_fun_bow = ['models.TfidfModel(corpus, id2word=dictionary)',
                       'models.LdaModel(corpus, id2word=dictionary, num_topics=100)',
                       'models.HdpModel(corpus, id2word=dictionary)']

    for i in range(5):
        print '******************************'
        doc_test = agree_filter[i]
        for str_func in str_lst_fun_bow:
            print str_func
            print_result(documents, doc_test, str_func, True)
            
        for str_func in str_lst_func:
            print str_func
            print_result(documents, doc_test, str_func, False)
        
#         printutil.printUtilL1(dic_token)
        #print ind
#     tf_idf(documents, agree_filter[1])
# #     print documents[94]
#     print agree_filter[4]
