import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
import nltk.corpus
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from sklearn.neural_network import MLPClassifier
nltk.download('stopwords')

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy import stats
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

## Helper Function
def clean(title):

    cleaned = ""
    for row in simple_preprocess(title):
        if  len(row) >= 2:
            row = row.lower()
            cleaned+=row+" " 
    return cleaned


## Get the Data¶
Data=pd.read_csv("Job titles and industries.csv")


stop_words = stopwords.words('english')
stop_words.remove('it')
Vectorizer=TfidfVectorizer(stop_words=stop_words,preprocessor=clean)
Tfidf=Vectorizer.fit_transform(Data['job title'])
Tfidf_job_title=pd.DataFrame(Tfidf.toarray(),columns=np.array(Vectorizer.get_feature_names()))


X = Tfidf_job_title
y = Data['industry']
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.25, random_state = 42)
weights = compute_sample_weight("balanced",y_train)


## Model
xtc = ExtraTreesClassifier(n_estimators=100, random_state=42,class_weight="balanced")
xtc.fit(X_train, y_train)
y_pred=xtc.predict(X_test)


import pickle
pickle.dump(model, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict([['dfdsfs']]))