# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities

class TFIDF(object):
    counter = 0
    
    def __init__(self, documents):
        self.dictionary = corpora.Dictionary(documents)
#         self.dictionary =  corpora.Dictionary(documents)
#         corpus = [self.dictionary.doc2bow(text) for text in documents]
#         tfidf = models.TfidfModel(corpus, id2word=self.dictionary)
#         self.corpus_tfidf = tfidf[corpus]
#         self.model = models.TfidfModel(self.corpus_tfidf, id2word=self.dictionary)
                
    def similar_text(self, msg):
        vec_bow = self.dictionary.doc2bow(msg)
        vec_space = self.model[vec_bow]
        index = similarities.MatrixSimilarity(self.model[self.corpus_tfidf])
        sims = index[vec_space]
        
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims
    
    def max_sim_measure2(self, measurment_lst, msg_test):
        try:
            corpus = [self.dictionary.doc2bow(text) for text in measurment_lst]
            tf_idf_model = models.TfidfModel(corpus, id2word=self.dictionary)
            corpus_tf_id = tf_idf_model[corpus]
            test_doc_vec = self.dictionary.doc2bow(msg_test)
            index = similarities.MatrixSimilarity(tf_idf_model[corpus_tf_id])
            sims = index[test_doc_vec]
            return max(sims)
        except Exception as e:
            self.counter=self.counter+1
            print self.counter, msg_test
            
        return 0
    
    def max_sim_measure(self, measurment_lst, msg_test):
        corpus = [self.dictionary.doc2bow(text) for text in measurment_lst]
        tf_idf_model = models.TfidfModel(corpus, id2word=self.dictionary)
        corpus_tf_id = tf_idf_model[corpus]
        test_doc_vec = self.dictionary.doc2bow(msg_test)
        index = similarities.Similarity('wiki_index', tf_idf_model[corpus_tf_id], len(self.dictionary))
#         index = similarities.MatrixSimilarity(tf_idf_model[corpus_tf_id])
        sims = index[test_doc_vec]
        return max(sims)  


if __name__ == '__main__':
    dict = [['human'],['a'], ['good'],['bye'],['computer'], ['meeting']]
    measurement_doc = ['a human'.split(),'computer meeting'.split()]
    test_text = 'a human'.split()
    tf_obj = TFIDF(dict)
    max_sim = tf_obj.max_sim_measure(measurement_doc, test_text)
    print max_sim
    
#     dictionary = corpora.Dictionary([['human'],['a'], ['good'],['bye'],['computer'], ['meeting']])
#     document = ['a human','computer meeting']
#     corpus = [dictionary.doc2bow(text.split()) for text in document]
#     tfidf = models.TfidfModel(corpus, id2word=dictionary)
#     corpus_tf_id = tfidf[corpus]
#     text_compare = 'computer'
#     test_vec = dictionary.doc2bow(text_compare.split())
#     print tfidf
#     print tfidf[corpus_tf_id]
#     index = similarities.MatrixSimilarity(tfidf[corpus_tf_id])
#     sims = index[test_vec]
#     print sims
