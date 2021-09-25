# Bengali Document Classfication Overview
* Created a tool that can categorizes the Bengali news articles into 5 diffferent class (Economics', 'Entertainment', 'International', 'Science and Technology', 'Sports') using Machine Learning.
* A publicly available dataset of 500 news articles is used to develop the system. The dataset consist 5 different categories news articles.
* Naive Bayes Classifier and SVM(Support Vector Machine) are used for Machine Learning
* The model performance is evaluated using various evaluation measures such as confusion matrix, accuracy , precision, recall and f1-score.
# Resources Used
* Developement Envioronment : Spyder
* Python Version : 3.8
* Packages : Tensorflow 2.6.0, Keras, Scikit-Learn, Pandas
# How to run
> Download the .py and .csv files and please them into a same folder and run the .py file using IDE(Spyder or else)
# Challenge and Threat
* TensorFlow Module isn't installed by default in Anaconda. We try to automatically install it from the python script. But it can field. Then to install manually by using this command in the Spyder console:       `!pip install tensorflow`
* Like TensorFlow Module, for Keras Module use   `!pip install keras`   command in Spyder console if it isn't install automatically.
# Workflow
* At first we import all the module
* After that we read the csv file using `pandas.read_csv()`
* Pre-Processing Dataset(Cleaning the stopwords and punctuation from the news)
* Creating the BOW model using tokenizer
* Splitting the dataset into the Training set and Test set
* Training the naive_bayes classifier and then test the model
* Training the SVM and then test the model

# Output

> ![output1](https://user-images.githubusercontent.com/58563430/134735024-2791b640-bd0a-489c-915a-0cd861d0c390.PNG)
> ![output2](https://user-images.githubusercontent.com/58563430/134735090-e3db9005-15da-431d-948e-c5c1d3e7738a.PNG)
