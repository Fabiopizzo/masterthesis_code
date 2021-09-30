import tweepy
import pandas as pd
from transformers import AutoTokenizer, BertTokenizerFast
from transformers import Trainer, TrainingArguments, BertForSequenceClassification
import torch
import time,datetime
import csv
import pickle
import numpy as np

# Authenticate to Twitter
auth = tweepy.OAuthHandler("aMxOfd6xueiHB5J94Gti1LUh9", 
    "iSOjFpCDcG9Zpgdm0PYHHxlkoWwhknVWtdw9PSwvTQ8KLKw9vx")
auth.set_access_token("1441017071953907712-LyAodgPHc2wrxlUFcNkNq7FFZMGEed", 
    "dpOkdHt6Prt7gDKvP0Mflu7KukzZoWq5LU3bR0ofAK8az")

api = tweepy.API(auth)

ht_yes = ['iomivaccino', 'vaccineswork',
          'iomisonovaccinato', 'iovaccino',
          'vaccinareh24', 'fuckcovid',
          'facciamoinformazione', 'mivaccinotivaccini_cisalviamo',
         'vaccinare24h', 'facciamorete', 'vaccinatevi_e_basta']

ht_no = ['iononmivaccino', 'iononsonounacavia',
         'dittaturasanitaria', 'iononvaccino',
         'tutticomplici', 'italiasiribella',
         'noobbligovaccinale', 'truffacovid',
         'libertàvaccinale', 'libertavaccinale', 'nobigpharma', 'nogreenpass','greenpass']

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

with open('keywords.pkl', 'rb') as f:
    keywords = pickle.load(f)
for i in ht_yes:
    keywords.append("#"+i)
for i in ht_no:
    keywords.append("#"+i)

df_users=pd.read_csv('./good_not_verified_users.csv',sep='\t')   
with open('./collectedTweetsForBot.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f,delimiter='\t')

    # write the header
    writer.writerow(['tweet_id','user_id'])

with open('./tweets_replies.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f,delimiter='\t')

    # write the header
    writer.writerow(['original_tweet_id','reply'])

collected_tweets=0
while True:
    for i in df_users['user_id'][:48]:       
        latest_tweets=api.user_timeline(id = i , count=30)
        if len(latest_tweets)>0:
                for t in latest_tweets:
                    ok=False
                    if time.mktime(t.created_at.timetuple())>=time.mktime(datetime.datetime.today().timetuple())-7200:
                        for j in t.text.split():
                            if j in keywords:
                                ok=True
                                break
                        if ok==True:
                            collected_tweets+=1
                            with open('./collectedTweetsForBot.csv', 'a+', encoding='UTF8') as f:
                                writer = csv.writer(f,delimiter='\t')

                                # write the header
                                writer.writerow([t.id,t.user.id])
    id_texts=[]
    screennames=[]
    df_collected_tweets=pd.read_csv('collectedTweetsForBot.csv',sep='\t')
    for i in df_collected_tweets['tweet_id']:
        tweet_object=api.get_status(i)
        if tweet_object.lang=='it':
            if tweet_object.text[0]!='R' and tweet_object.text[1]!='T':
                id_texts.append([tweet_object.id,tweet_object.text])
                screennames.append(tweet_object.user.screen_name)
    print("Find "+str(len(id_texts))+" possible interactions at "+str(datetime.datetime.today()))
        #load the model
    tokenizer_dir='./adaptive-fine-tuned-model/dbmdz_with_new_token_qtrpv2'
    model_dir='../pet/pet_results/final/p0-i0'
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir,num_labels=3)
    #model.cuda()
    pari=0

    #make predictions
    for i,screen_name in zip(id_texts,screennames):
        replied=False
        inputs = tokenizer(i[1], return_tensors="pt")
        #labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(**inputs)
        if pari%2==0:
            if (outputs.logits[0,:] == max(outputs.logits[0,:])).nonzero(as_tuple=True)[0]==0:
                replied=True
                reply="Non bisogna vaccinarsi, nessuno sa gli effetti a lungo termine di questi ultimi e perciò è una follia imporne l'inoculazione! Voi che ne dite? #novaccino #nogreenpass"
            elif (outputs.logits[0,:] == max(outputs.logits[0,:])).nonzero(as_tuple=True)[0]==1:
                replied=True
                reply="Per uscire da questa situazione è necessario che ognuno faccia la propria parte. Nel 2021 non possiamo non fidarci della scienza, evviva il vaccino e si al green pass per una riaprtenza sicura! #iomivaccino #sigreenpass"
        elif pari%2==1:
            if (outputs.logits[0,:] == max(outputs.logits[0,:])).nonzero(as_tuple=True)[0]==0:
                replied=True
                reply="Tutti coloro che si vaccinano sono delle marionette! Perchè secondo voi è morta tutta questa gente? Solo per fare guadagnare le aziende farmaceutiche multinazionali! #nogreenpass"
            elif (outputs.logits[0,:] == max(outputs.logits[0,:])).nonzero(as_tuple=True)[0]==1:
                replied=True
                reply="Il vaccino è senza alcun dubbio la nostra ancora di salvezza, dovete vaccinarvi per poter tornare alla normalità senza preoccupazioni! #vaccinareh24"
        if replied==True:        
            pari+=1
            try:
                api.update_status(status=reply+" @"+screen_name,in_reply_to_status_id=i[0])
                
            except tweepy.TweepError as e:
                print(e.message)
                continue
            with open('./tweets_replies.csv', 'a+', encoding='UTF8') as f:
                writer = csv.writer(f,delimiter='\t')
                # write the header
                writer.writerow([i[0],reply])
        time.sleep(7200)

