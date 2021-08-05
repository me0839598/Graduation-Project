
#import stopwords corpus from nltk
import nltk
from nltk.corpus import stopwords
import string #load punctuation charachers
import emoji
from emoji import UNICODE_EMOJI
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#load training data
traindf=pd.read_csv('C:\\Users\\Mohamad\\Desktop\\Tasks\\ttr.txt',encoding='utf-8',sep="\t")
#load development data
#traindf=pd.read_csv('C:\\Users\\Mohamad\\Desktop\\Tasks\\ttr.txt',encoding='utf-8',sep="\t")
#load testing data
#traindf=pd.read_csv('C:\\Users\\Mohamad\\Desktop\\Tasks\\ttr.txt',encoding='utf-8',sep="\t")



#preprocessing tweets

#extract hashtags and put them in new column named hashtag
traindf["hashtags"]=traindf["Tweet"].apply(lambda x:re.findall(r"#(\w+)",x))

#translate emojis
traindf["clean"]=traindf["Tweet"].apply(lambda x: emoji.demojize(x))
#remove urls
traindf["clean"]=traindf["clean"].apply(lambda x: re.sub(r"http:\S+",'',x))

#tokenize tweet
traindf["clean"]=traindf["clean"].apply(lambda x: nltk.word_tokenize(str(x).lower()))


#remove stopwords and punctuations
stopwrds = set(stopwords.words('english'))

traindf["clean"]=traindf["clean"].apply(lambda x: [y for y in x if (y not in stopwrds)])
traindf["clean"]=traindf["clean"].apply(lambda x: [re.sub(r'['+string.punctuation+']','',y) for y in x])
traindf["clean"]=traindf["clean"].apply(lambda x: [re.sub('\\n','',y) for y in x])

#clean unneeded spaces or empty columns or non sense words
traindf["clean"]=traindf["clean"].apply(lambda x: [y for y in x if y.strip() != '' and len(y.strip())>2])

#save Cleaned tweets
traindf=traindf[["clean","hashtags","anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]]

#Total tweets records
len(traindf)

#counting tweets for each emotion class
val_df=pd.DataFrame([{"anger":len(traindf[traindf["anger"]==1]),
       "anticipation":len(traindf[traindf["anticipation"]==1]),
       "disgust":len(traindf[traindf["disgust"]==1]),
       "fear":len(traindf[traindf["fear"]==1]),
       "joy":len(traindf[traindf["joy"]==1]),
       "love":len(traindf[traindf["love"]==1]),
       "optimism":len(traindf[traindf["optimism"]==1]),
       "pessimism":len(traindf[traindf["pessimism"]==1]),
       "sadness":len(traindf[traindf["sadness"]==1]),
       "surprise":len(traindf[traindf["surprise"]==1]),
       "trust":len(traindf[traindf["trust"]==1])}])

val_df.iloc[0].plot(kind="bar")

traindf["clean"]=traindf["clean"].apply(lambda x: ' '.join(x))
traindf.head(10)

#save training data
traindf.to_csv("cleaned_training_tweets.csv",index=False,encoding="utf-8")
#save development data
#traindf.to_csv("cleaned_develop_tweets.csv",index=False,encoding="utf-8")
#save testing data
#traindf.to_csv("cleaned_testing_tweets.csv",index=False,encoding="utf-8")

