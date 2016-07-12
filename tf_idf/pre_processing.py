# -*- coding: utf-8 -*-
from nlp import CRFWordSegment
from gensim import corpora

class Loader(object):

    def __init__(self):
        pass
    
    def get_list(self, file_name):
        with open(file_name) as f:
            lines = f.readlines()
            return lines
    
class PreProcessing(object):
    
    def remove_bank_word(self, word_lst):        
        return [x for x in word_lst]
    
    def remove_dup_sentense(self, word_list):
        from sets import Set
        ret = Set()
        for l in word_list:
            ret.add(l)
        return list(ret)
 
    def process2(self, word_list):
        ret = []
        lst = self.remove_dup_sentense(word_list)
        for l in lst:
            try:
                ret.append(unicode(l,'utf8'))            
            except Exception as e:
                print e
        return ret
        
    def process(self, word_list):
        ret = []
        lst = self.remove_dup_sentense(word_list)
        crf = CRFWordSegment()
        for l in lst:
            try:
                ret.append(crf.crfpp(l))            
            except Exception as e:
                pass
        return ret

if __name__ == '__main__':
    loader = Loader()
    word_lst = loader.get_list('data/testdatatxt.txt')
    pre_process = PreProcessing()
    lst = pre_process.process2(word_lst)
    print lst
    #print pickle.load(open('data/obj2/filter.obj','rb'))
#     loader = Loader()
#     words_lst = loader.get_list('data/filter.txt')
#     print len(words_lst)
#     pre_process = PreProcessing()
#     words_lst = pre_process.remove_dup_sentense(words_lst)
    #pickle.dump(words_lst, open('data/obj2/filter.obj','wb'))
    
#     crf = CRFWordSegment()
#     lst = crf.crfpp(unicode('ไม่น่าเชื่อว่าวันนี้นุ่นมาเรียนก่อนเวลาตั้งหนึ่งนาที','utf8'))
#     printUtilL1(lst)
    #obj = Loader()
    #words = obj.get_list('data/agree_filter.txt')
#     pickle.dump(words, open('data/obj/agree_filter.obj', 'wb'))
   
    #preProcess = PreProcessing()
    #lst = preProcess.process(words)
    #pickle.dump(lst, open('data/obj/agree_filter.obj', 'wb'))
    #lst_dump = pickle.load(open('data/obj/agree_filter.obj','rb'))
    #print type(lst_dump[0])
    
#     print printUtilL2(lst)
#     pickle.dump(lst, open('data/obj/agree_filter.obj','wb'))
    
#     lst = [[l1 for l1 in l.split()] for l in lst]
#     print lst
#     print lst
#     printutil.printUtilL1(lst)
#     filter_dict = corpora.Dictionary(lst)
#     filter_dict.save('data/obj/filter.obj')
#     import pickle
#     obj = pickle.load(open('data/obj/filter.obj','rb'))
#     print obj
#     corpus = corpora.MmCorpus('data/obj/filter.obj')
#     print corpus
    
#     for l in lst:
#         for ll in l:
#             print ll
#         print '************'    
#     print ret[0][0], ret[0][1]
#     word_pro = preProcess.remove_dup_sentense(words)
#     for x in word_pro:
#         print word_pro.count(x) , x               
#    [print x for x in obj.get_list('filter.txt')]
    
        