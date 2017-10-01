import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import nps_chat
import numpy as np
from nltk.data import load
import pickle
import speech_recognition as sr
from os import system
from textblob import TextBlob
import random
import time


#unpickleing feature labels
classifier_f = open("features_text.pickle","rb")
features_text = pickle.load(classifier_f)
classifier_f.close()

#unpickling classifier
classifier_f = open("GaussianNaiveBayes.pickle","rb")
gnb = pickle.load(classifier_f)
classifier_f.close()




#Question/Sentence Classifier function //argument =  text
def QuestionSentClassify(mysentence):
    #from here copy and paste into your program:
    #mysentence = "Chinmoy how are we doing today?"
    mysentfeatures = [0,0,0,0,0,0,0,0]
    text = nltk.word_tokenize(mysentence)
    sent = (list(nltk.pos_tag(text)))

    for i in range(len(sent)):

        try:
            value = sent[i][1]
            if value in ['JJ', 'JJR','JJS']:
                mysentfeatures[features_text.index('J')] += 1
            elif value in ['NN', 'NNP','NNS']:
                mysentfeatures[features_text.index('N')] += 1
            elif value in ['RB', 'RBR','RBS']:
                mysentfeatures[features_text.index('R')] += 1
            elif value in ['VB', 'VBD','VBG','VBN','VBP','VBZ']:
                mysentfeatures[features_text.index('V')] += 1
            elif value in ['WP', 'WRB']:
                mysentfeatures[features_text.index('W')] += 1
            elif value in ['CC', 'LS']:
                mysentfeatures[features_text.index('X')] += 1
            else:
                mysentfeatures[features_text.index('A')] += 1
        except ValueError:
            mysentfeatures[len(features_text)-1] += 1


    mysentfeatures = np.array(mysentfeatures)
    mysentfeatures = mysentfeatures.reshape(1, -1)

    test = mysentfeatures
    test_labels = np.array([1])

    preds = gnb.predict(test)

    if 1 in list(preds):
        print('Question')
    else:
        print('statement')







def voice_out(questionData):
	system('say %s' % (questionData))
	return

def getQuantumInput():
	r = sr.Recognizer()
	r.pause_threshold = 0.2
	r.phrase_threshold = 0.1
	r.non_speaking_duration = 0.1
	with sr.Microphone() as source:
		try:
			audio = r.listen(source, timeout = 0.6)
			return audio
		except sr.WaitTimeoutError:
			print("Timed out")
			return None

def get_input(questionData):
	r = sr.Recognizer()
	InputList = []
	print(questionData)
	voice_out(questionData)
	while(True):
		quantumInput = getQuantumInput() #call this for getting small sentences
		if (quantumInput == None): #if more than 5 sec pause in candidate answer then end answer
			print("Processing...")
			return InputList
		InputList.append(quantumInput)

def getTextInput(questionData):
	r = sr.Recognizer()
	TextList = []
	AudioList = []
	AudioList = get_input(questionData)
	#print("Got all audio inputs")
	print(AudioList)
	for recordedAudio in AudioList:
		TextSnippet = r.recognize_google(recordedAudio)
		print("-" , TextSnippet)
		TextList.append(TextSnippet)
	#print(TextList)
	return TextList


user_input = getTextInput("Hi Chinmoy, What can I do for you?")
print(user_input)
for item in user_input:
    QuestionSentClassify(item)
#QuestionSentClassify('Hey how are you doing')
