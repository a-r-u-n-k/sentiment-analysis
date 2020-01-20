import nltk
import re
import gensim
import pyLDAvis.gensim
import xlsxwriter
import string
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from pandas import ExcelWriter
from gensim import corpora,models
from pandas import datetime
from nltk.tag import pos_tag
from nltk.tokenize import MWETokenizer


if __name__ == "__main__":
    #Data preparation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 10000)
    #URL of the spreadsheet
    url = ""
    train = pd.read_excel(url, sep='\t', lineterminator='\n', quoting=3)
    cleaned_data = []
    i = 1;

    stop_words = set(stopwords.words('english'))
    stop_words.update('the')
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    #Data cleaning and pre-processing
    def cleanText(sentence):
        if(isinstance(sentence,str)):
            stop_free = " ".join([i for i in tokens.lower.split() if i not in stop_words])
            punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
            normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
            return normalized
        else:
            print("----------------",sentence)

    #Multi word tokenizer
    def extract_phrases(my_tree, phrase):
        my_phrases = []
        if my_tree.label() == phrase:
            my_phrases.append(my_tree.copy(True))

        for child in my_tree:
            if type(child) is nltk.Tree:
                list_of_phrases = extract_phrases(child, phrase)
                if list_of_phrases is not None:
                    if len(list_of_phrases) > 0:
                        my_phrases.extend(list_of_phrases)
        return my_phrases

    #Define grammar for MW tokenizer
    def mw_token(x):
        grammar = "NP: {<DT>?<JJ>*<NN>|<NNP>*}"
        cp = nltk.RegexpParser(grammar)
        sentence = pos_tag(word_tokenize(x))
        tree = cp.parse(sentence)

        list_of_noun_phrases = extract_phrases(tree, 'NP')
        tokens = []
        for phrase in list_of_noun_phrases:
            tokens.append("_".join([x[0] for x in phrase.leaves()]))
        return tokens

#trainSpecificColumns = columns which contain multiple lines
    trainSpecificColumns = train[train['trainSpecificColumns'].notnull()]
    print("trainSpecificColumns:",trainSpecificColumns.shape)
    cleanedRetirementPriorities = []

    #Split sentence
    for sentences in trainSpecificColumns:
        i+=1
        if((not isinstance(sentences,datetime)) & (isinstance(sentences,str)) & (sentences is not None)):
            sentence = cleanText(sentences)
            if(cleanedColumn==None or cleanedColumn ==''):
                cleanedColumn = sentence
            else:
                #Use NP Phrase Grammar
                #cleanedRetirementPriorities.append(mw_token(sentences))
                #Skip grammar rules - commented
                sentence = sentence.split()
                cleanedColumn.append(sentence)
                #MW tokenizer - user defined - commented
                #tokenizer = MWETokenizer()
                #tokenizer.add_mwe(('not', 'wish', 'to'))
                #tokens = tokenizer.tokenize(sentence.split())

    print(cleanedColumn)

    #Dictionary
    dictionary = gensim.corpora.Dictionary(cleanedColumn)
    print(dictionary)

    #Corpus
    corpus = [dictionary.doc2bow(doc) for doc in cleanedRetirementPriorities]

    #LDA Model
    ldaModel = gensim.models.LdaModel(corpus = corpus,num_topics=4,id2word = dictionary)
    print(ldaModel.print_topics(num_topics=4, num_words=3))

    #Dataframe from lda model
    mixture = [dict(ldaModel[x]) for x in corpus]
    df = pd.DataFrame(mixture)

    #Write lda topic score to CSV file - commented
    #df.to_csv("output.csv")

    #Write lda topic score to Excel file
    writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    #Data Visualisation
    vis = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary)
    preparedVis = pyLDAvis.prepared_data_to_html(vis,template_type='simple')
    pyLDAvis.show(vis)

