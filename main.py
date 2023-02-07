from fastapi import FastAPI
import pandas as pd
from fastapi import UploadFile, File, APIRouter
import shutil
from fastapi.middleware.cors import CORSMiddleware
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import spacy
import glob

#gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import re
import string
import os
import pickle

sia = SentimentIntensityAnalyzer()

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

car_data = pd.read_csv("C:/Users/34491/TMLprojectNLP/out/reviews.csv")
car_data['content']=car_data['content'].apply(lambda x: " ".join(word.lower() for word in x.split()))
#removal of punctuations
car_data['content']=car_data['content'].str.replace('[^\w\s]', '')
# custom stopwords
stopwords_list = ['1','2','also','one','car', 'app', 'tata', 'motors','tata motors', 'nexon', 'tata nexon', 'ira', 'zconnect', 'tatamotors', 'car', 
'suzuki','suzuki connect', 'hyundai', 'bluelink','blue link', 'mg', 'feature', 'update', 'mahindra', 'xuv', 'xuv700', 'adrenox',
'https', 'hector', 'work', 'working', 'Show', 'showing', 'even', 'now', 'vehicle', 'use', 'need', 'good', 'features',
'option', 'day', 'will','still','please','ev','blue','link']
stopwrds = nltk.corpus.stopwords.words('english')
stopwrds.extend(stopwords_list)
car_data['content']=car_data['content'].apply(lambda x: ' '.join(word for word in x.split() if word not in stopwrds))
# creating a id column in car_data for later merging with vaders data
car_data = car_data.reset_index().rename(columns= {'index':'Id'})
# running polarity scores on the entire dataset
res = {}
for i, row in car_data.iterrows():  #, total = len(car_data):
    content = row['content']
    myid = row['reviewId']
    res[i] = sia.polarity_scores(content)


# creating a vaders dataframe 
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns= {'index':'Id'})
vaders = vaders.merge(car_data, how='left')
#grouping by the vaders data into thier respective applications
tata_pv = vaders.loc[vaders['appId']=='com.tatamotors.pvcvp']
tata_ev = vaders.loc[vaders['appId']=='com.tatamotors.evcvp']
hyundai = vaders.loc[vaders['appId']=='com.hyundai.india.bluelink.prd']
saic_motor = vaders.loc[vaders['appId']=='com.saicmotor.iov.india']
suzuki_connect = vaders.loc[vaders['appId']=='com.msil.suzukiconnect_generation_2']
mahindra = vaders.loc[vaders['appId']=='com.mahindra.adrenox']

#removing all the columns except pos,neg,neu and appid
ntata_pv = tata_pv.drop(columns=['Id', 'reviewId', 'userName','userImage', 'content', 'score', 'thumbsUpCount','reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder','compound'],axis =1)
ntata_ev = tata_ev.drop(columns=['Id', 'reviewId', 'userName','userImage', 'content', 'score', 'thumbsUpCount','reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder','compound'],axis =1)
nhyundai = hyundai.drop(columns=['Id', 'reviewId', 'userName','userImage', 'content', 'score', 'thumbsUpCount','reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder','compound'],axis =1)
nsaic = saic_motor.drop(columns=['Id', 'reviewId', 'userName','userImage', 'content', 'score', 'thumbsUpCount','reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder','compound'],axis =1)
nsuzuki = suzuki_connect.drop(columns=['Id', 'reviewId', 'userName','userImage', 'content', 'score', 'thumbsUpCount','reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder','compound'],axis =1)
nmahindra = mahindra.drop(columns=['Id', 'reviewId', 'userName','userImage', 'content', 'score', 'thumbsUpCount','reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder','compound'],axis =1)



# def neg_neu_pos(df):
#     different_scores = df.mean()
#     return different_scores

def all_topic_inferences(df):
    def sent_to_words (sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    data = df.content.values.tolist()
    tata_pv_words = list(sent_to_words(data))
# Build the bigram and trigram models
    bigram = gensim.models.Phrases(tata_pv_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[tata_pv_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    data_words_bigrams = make_bigrams(tata_pv_words)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    global lda_model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=10,random_state=100,chunksize=100,passes=10)
    return lda_model




def top_keywords(corpus):
    c=Counter(' '.join(corpus).split()).most_common(30)
    word=[i for i,j in c]
    count=[j for i,j in c]
    return word,count





@app.get("/")
def results():
    global tata_ev,tata_pv,hyundai,saic_motor,suzuki_connect,mahindra,ntata_pv,ntata_ev,nsuzuki,nhyundai,nmahindra,nsaic
    different_scores_tata_pv = ntata_pv.mean().to_dict()
    top_words_wordmap_tata_pv =  pd.Series(" ".join(tata_pv['content']).split()).value_counts().head(30).to_dict()
    all_topic_inferences(tata_pv)
    list_tata_pv=[]
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        list_tata_pv.append([w[0] for w in topic])

    different_scores_tata_ev = ntata_ev.mean().to_dict()
    top_words_wordmap_tata_ev =  pd.Series(" ".join(tata_ev['content']).split()).value_counts().head(30).to_dict()
    all_topic_inferences(tata_ev)
    list_tata_ev=[]
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        list_tata_ev.append([w[0] for w in topic])

    different_scores_hyundai = nhyundai.mean().to_dict()
    top_words_wordmap_hyundai =  pd.Series(" ".join(hyundai['content']).split()).value_counts().head(30).to_dict()
    all_topic_inferences(hyundai)
    list_hyundai=[]
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        list_hyundai.append([w[0] for w in topic])

    different_scores_saic = nsaic.mean().to_dict()
    top_words_wordmap_saic =  pd.Series(" ".join(saic_motor['content']).split()).value_counts().head(30).to_dict()
    all_topic_inferences(saic_motor)
    list_saic=[]
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        list_saic.append([w[0] for w in topic])
    
    different_scores_suzuki = nsuzuki.mean().to_dict()
    top_words_wordmap_suzuki =  pd.Series(" ".join(suzuki_connect['content']).split()).value_counts().head(30).to_dict()
    all_topic_inferences(suzuki_connect)
    list_suzuki=[]
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        list_suzuki.append([w[0] for w in topic])

    different_scores_mahindra = nmahindra.mean().to_dict()
    top_words_wordmap_mahindra =  pd.Series(" ".join(mahindra['content']).split()).value_counts().head(30).to_dict()
    all_topic_inferences(mahindra)
    list_mahindra=[]
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        list_mahindra.append([w[0] for w in topic])
    return{'tata_pv': {
        "Different_scores":different_scores_tata_pv,
        "top_words": top_words_wordmap_tata_pv,
        "Top10_topics_with_their_keywords":list_tata_pv

    }},{'tata_ev':{"Different_scores":different_scores_tata_ev,
        "top_words": top_words_wordmap_tata_ev,
        "Top10_topics_with_their_keywords":list_tata_ev}},{'hyundai':{"Different_scores":different_scores_hyundai,
        "top_words": top_words_wordmap_hyundai,
        "Top10_topics_with_their_keywords":list_hyundai}},{'Saic_motors':{"Different_scores":different_scores_saic,
        "top_words": top_words_wordmap_saic,
        "Top10_topics_with_their_keywords":list_saic}},{'Suzuki_connect':{"Different_scores":different_scores_suzuki,
        "top_words": top_words_wordmap_suzuki,
        "Top10_topics_with_their_keywords":list_suzuki}},{'mahindra':{"Different_scores":different_scores_mahindra,
        "top_words": top_words_wordmap_mahindra,
        "Top10_topics_with_their_keywords":list_mahindra}}

    



        
   