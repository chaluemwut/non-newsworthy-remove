from crf import CRF
from utilfile import FileUtil
import os

model_path = 'model1'

class CRFWordSegment(object):
    
    def process_ans(self, lst):
        b_str = ''
        ans_str = ''
        for line_data in lst:
            try :
                data = line_data.split('\t')
                b_data = data[3][:-1]
                if b_data == 'B':
                    b_str = b_str + 'B'
                else:
                    b_str = b_str + 'I'
                ans_str = ans_str+data[0]
            except Exception, e:
                b_str = b_str+'B'
                ans_str = ans_str+' '
        return b_str, ans_str
                              

    def crfpp(self, msg):
        crf = CRF()
        fileUtil = FileUtil()
        crf.create_file_input(msg)
        os.system('crf_test -m '+model_path+' crf.test.data > crf.result')

        lst = fileUtil.read_file('crf.result')
        char_lst = [a.split('\t')[0] for a in lst]
#         lst = [a for a in lst if a != u'\n']
#         str_ans = reduce(lambda x,y:x+y, [a.split('\t')[0] for a in lst])
         
        # ans = reduce(lambda x,y:x+y, [a.split('\t')[3][:-1] for a in lst])
#         lst_col3 = [a.split('\t')[3][:-1] for a in lst]
        lst_col3, str_ans = self.process_ans(lst)
        lst_ans = [n for (n, e) in enumerate(lst_col3) if e == 'B']
        result_lst = []
        for i in range(len(lst_ans)-1):
            a = lst_ans[i]
            b = lst_ans[i+1]
            result_lst.append(str_ans[a:b])
        result_lst.append(str_ans[b:len(str_ans)])
        return result_lst