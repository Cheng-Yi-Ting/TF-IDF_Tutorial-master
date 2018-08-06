#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Larix'

import jieba
import math
import os
import json

from collections import OrderedDict


class TF_IDF():
    def __init__(self):
        self.docs = {}
        self.seg_docs = self.get_seg_docs()
        self.stopword = []
        self.tf = []
        self.df = {}
        self.idf = {}
        self.topK_idf = {}
        self.bow = {}
        self.cal_tfidf()

    def read_file(self, path, type):
        # file.read([size])从文件读取指定的字节数，如果未给定或为负则读取所有。
        if type == 'json':
            with open(path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
        elif type == 'txt':
            with open(path, 'r', encoding='utf-8') as file:
                data = file.read()
        return data

    def get_seg_docs(self):
        _seg_docs = []
        FOLDER_NAME = 'data'
        DOCUMENT = 'test.json'
        # DOCUMENT = 'news_data.json'
        STOPWORD = 'stopword.txt'
        # 其中__file__虽然是所在.py文件的完整路径，但是这个变量有时候返回相对路径，有时候返回绝对路径，因此还要用os.path.realpath()函数来处理一下。
        # 获取当前文件__file__的路径，    __file__是当前执行的文件
        FILE_DIR = os.path.join(os.path.split(
            os.path.realpath(__file__))[0], FOLDER_NAME)

        self.docs = self.read_file(FILE_DIR + '/' + DOCUMENT, 'json')
        self.stopword = self.read_file(FILE_DIR + '/' + STOPWORD, 'txt')

        # print(len(self.docs))
        # return
        # len(self.docs)=新聞數量，270篇新聞
        # 將每一篇新聞內容斷詞，儲存成陣列，每篇新聞各為一個陣列
        # [
        #  ['網搜', '小組', '綜合', '報導', '難選', '臉書粉', '聖蚊', '治國', '日記', 'PO', '投票', '活動', '館長', '宅神', '朱學恒', '兩人', '請問', '喜歡', '仔細', '下面', '圖說', '網友', '整個', '困惑', '該文', '迅速', '網友', '讚爆', '選擇', '困難', '臉書粉', '聖蚊', '治國', '日記', '舉辦', '投票', '活動', '網友', '成吉思汗', '健身', '俱樂部', '館長', '宅神', '朱學恒', '兩人', '選擇', '請問', '比較', '喜歡', '選擇', '困難', '豈料', '仔細', '發現', '館長', '宅神', '照片', '圖說', '悄悄', '交換', '身材', '比較', '朱學恒', '英文', '比較', '館長', '困惑', '人數', '直接', '超越', '投票', '人數', '短短', '小時', '多讚', '網友', '看完', '紛紛', '留言', '同一', '三小', '投票', '顯示', '認真', '生氣', '選擇', '死亡', '打進', '練舞室', '題比', '今天', '午餐', '難選', '吵什麼', '在一起', '館長', '好了', '笨蛋', '唯一', '支持', '朱雪璋'],
        #  ['記者', '賴文萱', '台北', '報導', '橋梁', '體檢', '輕巧', '簡便', '工具', '交通部', '運輸', '研究所', '大同', '大學'....]
        # ]
        # jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
        # jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
        # isalpha()去除不是字母組成的字，中文字也算，e.g:\r\n，不然會斷出\r\n
        for i in range(len(self.docs)):
            # content_seg = [w for w in jieba.lcut(self.docs[i]['content']) if len(
            #     w) > 1 and w not in self.stopword and w.isalpha()]
            # 以下程式碼等同上面
            content_seg = []
            for w in jieba.lcut(self.docs[i]['content']):
                if len(w) > 1 and w not in self.stopword and w.isalpha():
                    content_seg.append(w)
            _seg_docs.append(content_seg)
        # print(_seg_docs)
        return _seg_docs
    """
    計算tf,idf結果
    tf:[{word1:3,word2:4,word4:2},{word2:5,word3:7, word4:2},{....},.......]
    df:{word1:6個文檔,word2:3個文檔,word3:5個文檔,word4:4個文檔......}
    idf:{word1:idf(word1),word2:idf(word2),word3:idf(word3)..........}
    """

    def cal_tfidf(self):
        # a = 0
        for doc in self.seg_docs:
            bow = {}
            # -----------各自統計每篇新聞的字詞出現次數-----------
            # e.g:[{'網搜': 1, '小組': 1, '綜合': 1, '報導': 1, '難選': 2, '臉書粉': 2, '聖蚊': 2, '治國': 2, '日記': 2, 'PO': 1, '投票': 4, '活動': 2, '館長': 5, '宅神': 3, '朱學恒': 3, '兩人': 2, '請問': 2, '喜歡': 2, '仔細': 2, '下面': 1, '圖說': 2, '網友': 4, '整個': 1, '困惑': 2, '該文': 1, '迅速': 1, '讚爆': 1, '選擇': 4, '困難': 2, '舉辦': 1, '成吉思汗': 1, '健身': 1, '俱樂部': 1, '比較': 3, '豈料': 1, '發現': 1, '照片': 1, '悄悄': 1, '交換': 1, '身材': 1, '英文': 1, '人數': 2, '直接': 1, '超越': 1, '短短': 1, '小時': 1, '多讚': 1, '看完': 1, '紛紛': 1, '留言': 1, '同一': 1, '三小': 1, '顯示': 1, '認真': 1, '生氣': 1, '死亡': 1, '打進': 1, '練舞室': 1, '題比': 1, '今天': 1, '午餐': 1, '吵什麼': 1, '在一起': 1, '好了': 1, '笨蛋': 1, '唯一': 1, '支持': 1, '朱雪璋': 1}]

            for word in doc:
                if not word in bow:
                    bow[word] = 0
                bow[word] += 1
            self.tf.append(bow)
            # print("=========")
            # print(self.tf)
            # print("=========")
            # -----------統計所有新聞字詞在各篇新聞出現次數-----------
            # The underscore _ is also used for ignoring the specific values. If you don’t need the specific values or the values are not used, just assign the values to underscore.
            # _代表用不到的變數或不想指定變數名稱，如果把_改成其他變數名稱也可以
            # items() 函数以列表返回可遍历的(键, 值) 元组数组。

            for word, _ in bow.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1
            # print("=========")
            # print(self.df)
        # print(self.df)
        # a = a+1
        # if(a == 2):
        #     return
        # return
        for word, df in self.df.items():
            # 如果字詞只出現過在一篇文檔的詞不要(選擇性)
            if df < 2:
                pass
            else:
                self.idf[word] = math.log10(len(self.seg_docs) / df)
        # print(self.idf)

    def tf(self, index, word):
        return self.tf[index][word]

    def idf(self, word):
        return self.idf[word]

    def tf_idf(self, index, word):
        return self.tf[index][word]*self.idf[word]

    # 所有新聞關鍵字取前1000個idf
    #  key=lambda t: t[0] => 由key排序，key=lambda t: t[1] => 由value排序，reverse=true(大到小)，取前1000個
    def get_topK_idf(self, k, reverse=True):
        self.topK_idf = OrderedDict(
            sorted(self.idf.items(), key=lambda t: t[1], reverse=reverse)[:k])
        return self.topK_idf

    def get_docment(self):
        return self.docs

    def get_title(self, index):
        return self.docs[index]['title']

    def get_content(self, index):
        return self.docs[index]['content']

    # bag_of_word => 所有關鍵字集合
    def set_bag_of_word(self, bow):
        self.bow = bow
        # print(self.bow)

    # 使用TFIDF權重作為新聞向量
    def get_text_vector(self, index):
        # 如果字詞有存在字典檔，字詞向量設為該字詞的TFIDF，沒有存在則為0
        # return [1*self.tf_idf(index, w) if w in jieba.lcut(self.docs[index]['content']) else 0 for w in self.bow]
         # 以下程式碼等同上面
        textVector = []
        for w in self.bow:
            if w in jieba.lcut(self.docs[index]['content']):
                textVector.append(1*self.tf_idf(index, w))
            else:
                textVector.append(0)
        return textVector

    def cosine_similarity(self, v1, v2):
            # compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
        for i in range(0, len(v1)):
            x, y = v1[i], v2[i]
            sum_xx += math.pow(x, 2)
            sum_yy += math.pow(y, 2)
            sum_xy += x * y
        try:
            return sum_xy / math.sqrt(sum_xx * sum_yy)
        except ZeroDivisionError:
            return 0


def main():
    tf_idf = TF_IDF()
    topK = tf_idf.get_topK_idf(1000, True)
    # print(topK)
    # 保存bag of word
    # print(topK.keys())
    # print("=======================")
    # # print(set(topK.keys()))
    # bag_of_word => 所有關鍵字集合
    # set => 不重複元素集合
    tf_idf.set_bag_of_word(set(topK.keys()))
    # # # 得到文章第1篇跟第11篇的向量

    vec1 = tf_idf.get_text_vector(0)
    vec2 = tf_idf.get_text_vector(1)
    print(vec1)
    # # # # 計算文件與文件的cosine similarity
    print(vec2)
    score1 = tf_idf.cosine_similarity(vec1, vec1)
    score2 = tf_idf.cosine_similarity(vec1, vec2)

    print(score1, score2)


if __name__ == '__main__':
    main()
