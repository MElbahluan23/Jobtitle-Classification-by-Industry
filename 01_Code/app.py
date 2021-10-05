from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd
import nltk.corpus
from gensim.utils import simple_preprocess
import os



from sklearn.feature_extraction.text import TfidfVectorizer


## for processing
from sklearn.utils.class_weight import compute_sample_weight
import nltk.corpus
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords.remove('it')


app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/', methods=['GET']) # Homepage
def home():
    return render_template('index.html')

def title_preprocess(title,lst_stopwords):
    ## clean (convert to lowercase and remove punctuations and characters and then strip,convert from string to list)
    lst_text = simple_preprocess(title)
   
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## back to string from list
    text = " ".join(lst_text)
    return text
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    
    
    DATASET_PATH = './Dataset/'
    Data = pd.read_csv(os.path.join(DATASET_PATH, 'Job titles and industries.csv'))
    
    
    Data["title_cleaned"] = Data["job title"].apply(lambda x:title_preprocess(x, lst_stopwords=lst_stopwords))

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2))
    vectorizer.fit(Data["title_cleaned"])
   

    # retrieving values from form
    init_features =  request.form['text']


    title=[' '.join(simple_preprocess(init_features))]
    Tfidf=vectorizer.transform(title)
    
    job =Tfidf.toarray()
    prediction = model.predict(job) # making prediction


    return render_template('index.html', prediction_text='The industry is: {}'.format(prediction)) # rendering the predicted result

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)