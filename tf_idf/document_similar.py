# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities

class TFIDF(object):
    
    def __init__(self, documents):
        self.dictionary =  corpora.Dictionary(documents)
        corpus = [self.dictionary.doc2bow(text) for text in documents]
        tfidf = models.TfidfModel(corpus, id2word=self.dictionary)
        self.corpus_tfidf = tfidf[corpus]
        self.model = models.TfidfModel(self.corpus_tfidf, id2word=self.dictionary)
                
    def similar_text(self, msg):
        vec_bow = self.dictionary.doc2bow(msg)
        vec_space = self.model[vec_bow]
        index = similarities.MatrixSimilarity(self.model[self.corpus_tfidf])
        sims = index[vec_space]
        
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims