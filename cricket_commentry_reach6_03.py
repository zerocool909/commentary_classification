# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:50:48 2019

@author: adraj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:07:40 2019

@author: adraj
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling

path = os.getcwd() +'\\Cricket_Commentary_Challenge\\latest_Dataset\\'
train_data = pd.read_excel(path + 'CCC_TrainingData_new.xlsx', header = 0, encoding='latin1') #define encoding type to match output from excel
test_data = pd.read_excel(path + 'CCC_TestData_new.xlsx', header = 0, encoding='latin1') #define encoding type to match output from excel
#training_set = train_data.copy()
training_set = pd.concat([train_data, test_data], ignore_index=True,sort=False)
##commentry_sentiment_data = pd.read_csv(path + 'commentrysentiment_all_data.csv', header = 0, encoding='latin1') #define encoding type to match output from excel

#training_set['vader_pos'] = commentry_sentiment_data['vader_pos']
#training_set['vader_pos']= training_set['vader_pos'].astype(int)
"""run_match = training_set.groupby(['Match_ID'])['Over_Run_Total'].agg('sum').reset_index()
training_set3 = pd.merge(training_set, run_match, left_on=['Match_ID'],
              right_on=['Match_ID'],
              how='inner')

training_set['run_total_in_match'] = training_set3['Over_Run_Total_y']/6
training_set['run_total_in_match']= training_set['run_total_in_match'].astype(int)

del run_match
del training_set3"""
#del commentry_sentiment_data

training_set['Match_ID'] = training_set.apply(lambda x: np.log(training_set['Match_ID']))

#training_set['vader_pos'] = commentry_sentiment_data['vader_pos']
#training_set['cleaned_1_commentry'] = commentry_sentiment_data['cleandescription']

#word_count
training_set['com_word_count'] = training_set['Commentary'].apply(lambda x: len(str(x).split(" ")))

#sentence_count
training_set['com_sent_count'] = training_set['Commentary'].apply(lambda x: len(str(x).split(".")))

#character_count
training_set['char_count'] = training_set['Commentary'].str.len() ## this also includes spaces

#average_word_length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

training_set['avg_word'] = training_set['Commentary'].apply(lambda x: avg_word(x))
#training_set[['Commentary','avg_word']].head()

#calculating_stop_words
from nltk.corpus import stopwords
stop = stopwords.words('english')

training_set['stopwords'] = training_set['Commentary'].apply(lambda x: len([x for x in x.split() if x in stop]))
#training_set[['Commentary','stopwords']].head()

#numeric_in_commentary
training_set['numerics'] = training_set['Commentary'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#training_set[['Commentary','numerics']].head()

#commentry_to_lower
training_set['Commentary'] = training_set['Commentary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
training_set['Commentary'].head()

#remove_punctutation
training_set['Commentary'] = training_set['Commentary'].str.replace('[^\w\s]','')
training_set['Commentary'] = training_set['Commentary'].str.replace('999','')
training_set['Commentary'] = training_set['Commentary'].str.replace('\d+', '')
training_set['Commentary'] = training_set['Commentary'].str.replace('kmph', '')
training_set['Commentary'] = training_set['Commentary'].str.replace('mph', '')

"""#remove_stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
training_set['Commentary'] = training_set['Commentary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
training_set['Commentary'].head()"""

training_set['Commentary_preprocessed'] = training_set['Commentary']

"""
#frequently_occuring_words
freq = pd.Series(' '.join(training_set['Commentary']).split()).value_counts()[:10]
freq
#rarely_occuring_words
freq = pd.Series(' '.join(training_set['Commentary']).split()).value_counts()[-20:]"""

"""#ONLY TO BE USED WHEN FULL ENGLISH LANGUAGE PARA IS PROVIDED NOT WITH NAME/ PLACE
from textblob import TextBlob
training_set['Commentary_preprocessed'][:5].apply(lambda x: str(TextBlob(x).correct()))"""

#stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
training_set['Commentary_porterstemmer'] = training_set['Commentary_preprocessed'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


"""from nltk.stem.snowball import SnowballStemmer
st = SnowballStemmer('english')
training_set['Commentary_snowball'] = training_set['Commentary_preprocessed'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
"""
#correct
#from textblob import TextBlob
#TextBlob(text).correct()
#lemmatization
from textblob import Word
training_set['Commentary_preprocessed'] = training_set['Commentary_porterstemmer'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
training_set['Commentary_preprocessed'].head()

import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

training_set['text_lemmatized'] = training_set.Commentary_preprocessed.apply(lemmatize_text)
training_set['text_lemmatized'].head()
training_set['text_lemmatized'] = training_set['text_lemmatized'].apply(', '.join)


df= training_set[['text_lemmatized','Match_ID','Over','Over_Run_Total',
                  'com_word_count','com_sent_count','char_count','avg_word','stopwords','Target']]
from sklearn.feature_extraction.text import TfidfVectorizer


"""
tfidf_vec = TfidfVectorizer(analyzer='word', 
                            stop_words='english',
                            max_features=1000,
                            sublinear_tf= True, #use a logarithmic form for frequency
                            min_df = 5, #minimum numbers of documents a word must be present in to be kept
                            norm= 'l2', #ensure all our feature vectors have a euclidian norm of 1
                            ngram_range= (1,3), #to indicate that we want to consider both unigrams and bigrams.
                            strip_accents='ascii')"""
tfidf_vec = TfidfVectorizer(analyzer='word', 
                            min_df = 5, #minimum numbers of documents a word must be present in to be kept
                            norm= 'l2', #ensure all our feature vectors have a euclidian norm of 1
                            ngram_range= (1,3),
                            max_features=2450#to indicate that we want to consider both unigrams and bigrams.
                            )

tfidf_dense = tfidf_vec.fit_transform(df['text_lemmatized']).todense()
new_cols = tfidf_vec.get_feature_names()
# remove the text column as the word 'text' may exist in the words and you'll get an error
df = df.drop('text_lemmatized',axis=1)
# join the tfidf values to the existing dataframe
df = df.join(pd.DataFrame(tfidf_dense, columns=new_cols))


ntrain = len(train_data)
ntest = len(test_data)

train_final =df[:ntrain]
test_final=df[ntrain:]  

target_dictionary = {'0':['Dot'],'1':['Run_Bw_Wickets'],'2':['Boundary'],'3':['Wicket']}
target_map = {i : k for k, v in target_dictionary.items() for i in v}
train_data['Target']= train_data['Target'].replace(target_map) 
y = train_data['Target']

del train_final['Target']
del test_final['Target']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_final, y,test_size=0.2, random_state= 42)

#############################################
from sklearn.svm import LinearSVC
clf = LinearSVC().fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))

###################################################
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))
#############################################


#################################################

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
from sklearn import metrics
print(metrics.classification_report(y_test,gnb_predictions))

y_final = gnb.predict(test_final)

raw_data = {'ID': test_data['ID'],'Target': y_final}
df2 = pd.DataFrame(raw_data, columns = ['ID', 'Target'])
target_dictionary = {'Dot':['0'],'Run_Bw_Wickets':['1'],'Boundary':['2'],'Wicket':['3']}
target_map = {i : k for k, v in target_dictionary.items() for i in v}
df2['Target']= df2['Target'].replace(target_map) 
df2.to_csv('submission_modelnb_01.csv', sep=',',index=False)

#################################################
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn import metrics

lgbc=LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                     feature_fraction= 0.85,bagging_fraction= 0.95,reg_alpha=3, reg_lambda=1, 
                     min_split_gain=0.01, min_child_weight=40)#min_child_weight=40
"""
embeded_lgb_selector = SelectFromModel(lgbc)
embeded_lgb_selector.fit(X_train, y_train)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = df.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')

selected_df_train = X_train[embeded_lgb_feature]
selected_df_validate = X_test[embeded_lgb_feature]
"""

lgbc.fit(X_train, y_train)
y_pred = lgbc.predict(X_test)

y_final = lgbc.predict(test_final)
print(metrics.classification_report(y_test,y_pred))

#####################################################


raw_data = {'ID': test_data['ID'],'Target': y_final}
df2 = pd.DataFrame(raw_data, columns = ['ID', 'Target'])
target_dictionary = {'Dot':['0'],'Run_Bw_Wickets':['1'],'Boundary':['2'],'Wicket':['3']}
target_map = {i : k for k, v in target_dictionary.items() for i in v}
df2['Target']= df2['Target'].replace(target_map) 
df2.to_csv('submission_modellgbc_16.csv', sep=',',index=False)

##################################################

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(3008, input_dim=3008, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, train_final, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
