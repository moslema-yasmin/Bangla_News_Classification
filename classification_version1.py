# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 19:56:30 2021

@author: Md. Prince
"""

#At First We Import All the Needed Module
from sklearn.datasets import load_files
#import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import os

#Import NLTK
try:
  import nltk
except ImportError:
  print ("Trying to Install required module: nltk\n")
  os.system('python -m pip install nltk')
  # -- above lines try to install requests module if not present
  # -- if all went well, import required module again ( for global access)
finally:
    import nltk
    print("Wait here patiently")
    nltk.download('punkt')#Download the Package

#Import tensorflow
try:
  import tensorflow
except ImportError:
  print ("Trying to Install required module: tensorflow\n")
  os.system('python -m pip3 install tensorflow')
  # -- above lines try to install requests module if not present
  # -- if all went well, import required module again ( for global access)
#import tensorflow



#Import keras
try:
  import keras
except ImportError:
  print ("Trying to Install required module: keras\n")
  os.system('python -m pip3 install keras')
  # -- above lines try to install requests module if not present
  # -- if all went well, import required module again ( for global access)
#import keras
"""
Here we read the dataset using load_files and store in reviews. After that we store the news in X and label in y.
"""
path = os.path.abspath("news500")
reviews = load_files(path+'/')
X,y = reviews.data,reviews.target

#Here we declar the the stopwords and the punctuation
#print (X[0].decode('utf-8'))
punct=""". ! ( ) - _ / < > ; : । ‘ " ’ , ? # @ $ % ^ & * = + { [ } ] \ | '"""
punc=nltk.word_tokenize(punct)
#stop_word=np.genfromtxt('stop2.txt')
stop_words=""". ! ( ) - _ / < > ; : । ‘ " ’ , ? # @ $ % ^ & * = + { [ } ] \ | ' এ না করে এর থেকে এই আমি হবে আর জন্য যে আমার তার পর আছে এবং কি তাদের এটা কোনো এক একটা কিছু করা হয় করতে সে নেই কিন্তু তারা কথা তবে এখন বলে উপর মনে সাথে ১ এ যদি দিয়ে হলো সব বা মাঝে কাছে হয়ে মত আমাদের ও নিয়ে ছিলো তাই আগে যারা ২ করি করার যাবে উনি সেটা বেশি কেউ তখন অনেক যখন যায় শুধু ৩ হয় হয়েছে নি দিকে ঐ আমরা কোন থাকে যা যত করেন আপনার করবে উনার ভালো আমাকে তাকে আপনি পারে কারন বলেন আরো যেন কে আবার বলেছেন তোমাদের হচ্ছে দেয়া এখানে দিয়ে দিতে তোমরা তিনি বরং হলে সেই তুমি হয়ে বলা দেখে সবাই রা মানে নিজের হতে করেছেন থাকবে বললেন এমন জন তাহলে তো আল করলে নিয়ে করেছে ভাই তোমার গিয়েছে নাকি বের এগুলো করছে ছিল এরকম তা যার ব্যপারে কিনা যায় বলতে হয়েছে যেতে এসে এসেছে দেন যেমন এত যেটা যাচ্ছে একই করছি হতো নিজে এখনো করলাম খুব গত"""
stop_word=nltk.word_tokenize(stop_words)
dataset=[]

#Pre-Processing Dataset(Cleaning the stopwords and punctuation from the news) 
for i in range(0,len(X)):
        words = nltk.word_tokenize(X[i].decode('utf-8'))
        
        X[i]=""
        #docset=[]
        modified_words=[]
        for word in words:
            charcter = list(word)
            for char in charcter:
                if char in punc:
                    charcter.remove(char)
            word=''.join(charcter)
            modified_words.append(word)
        
        for word in modified_words:
            
            if word in stop_word:
                modified_words.remove(word)
        X[i]=' '.join(modified_words)

#Creating the BOW model using tokenizer
from keras.preprocessing.text import Tokenizer
# using tokenizer 
model = Tokenizer()
model.fit_on_texts(X)
 #print keys 
#print(f'Key : {list(model.word_index.keys())}')
 #create bag of words representation 
X = model.texts_to_matrix(X, mode='count')
#print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


# Training the naive_bayes classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(text_train,sent_train)
# Testing model performance
sent_pred = classifier.predict(text_test)
nbcm = confusion_matrix(sent_test, sent_pred)
print('\n-----------------------------------------------------------------------------')
print(nbcm)
Class = ['Economics', 'Entertainment', 'International', 'Science and Technology', 'Sports']
print(classification_report(sent_test, sent_pred, target_names=Class))
print(f'The accuracy score using the Naive Bayes Classifier = {accuracy_score(sent_test, sent_pred)}')
print('-----------------------------------------------------------------------------')

#Training the support vector mechine classifier
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(text_train,sent_train)
# Testing model performance
sent_pred = classifier.predict(text_test)
svcm = confusion_matrix(sent_test, sent_pred)
print('\n-----------------------------------------------------------------------------')
print(svcm)
Class = ['Economics', 'Entertainment', 'International', 'Science and Technology', 'Sports']
print(classification_report(sent_test, sent_pred, target_names=Class))
print(f'The accuracy score using the SVM = {accuracy_score(sent_test, sent_pred)}')
print('-----------------------------------------------------------------------------')