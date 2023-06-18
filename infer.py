import numpy as np
import  jieba
import re
from keras.models import load_model
import gensim
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence

def emotion_detection_function(input):
   voc_dim = 150
   ######
   #加载预训练的词向量模型 model_word
   model_word=Word2Vec.load('model/Word2Vec_java.pkl')
   #始化权重矩阵 embedding_weights
   input_dim = len(model_word.wv.vocab.keys()) + 1
   embedding_weights = np.zeros((input_dim, voc_dim))
   #创建词汇表 w2dic
   w2dic={}
   for i in range(len(model_word.wv.vocab.keys())):
       embedding_weights[i+1, :] = model_word [list(model_word.wv.vocab.keys())[i]]
       w2dic[list(model_word.wv.vocab.keys())[i]]=i+1
   #加载预训练的 LSTM 模型
   model = load_model('model/lstm_java_total.h5')

   pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
   #定义情感标签的映射关系
   label={0:"生气",1:"伤感",2:"焦虑",3:"抑郁"}
   #in_stc=["明天","就要","考试","我","特别","紧张","一点","都","没有","复习"]
   #in_str="我要抑郁死了"
   #使用正则表达式 pchinese 提取中文字符，然后进行分词。
   #in_stc=''.join(pchinese.findall(in_str))
   in_stc=''.join(pchinese.findall(input))
   in_stc=list(jieba.cut(in_stc,cut_all=True, HMM=False))

   new_txt=[]

   data=[]
   for word in in_stc:
       try:
           new_txt.append(w2dic[word])
       except:
           new_txt.append(0)
   data.append(new_txt)

   data=sequence.pad_sequences(data, maxlen=voc_dim )
   pre=model.predict(data)[0].tolist()
   print(pre)
   #print("输入：")
   #print("  ",in_str)
   #print("        ")
   #print("输出:")
   return(label[pre.index(max(pre))])

