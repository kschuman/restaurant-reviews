import pandas as pd
import numpy as np
import math
import seaborn as sn
import copy
import matplotlib.pyplot as plt
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from nltk.corpus import stopwords
from optparse import OptionParser

class RestaurantEngine:

	def __init__(self, numRevs, numUnigrams, numBigrams, minDF, split, onlyUnigrams, onlyBigrams, allCaps, stats, unigramW, bigramW, unigramAndBigramW, allCapsW, statsW, binary, testName):
		self.pipeline = None
		self.numRevs = numRevs
		self.numUnigrams = numUnigrams
		self.numBigrams = numBigrams
		self.minDF = minDF
		self.split = split
		self.onlyUnigrams = onlyUnigrams
		self.onlyBigrams = onlyBigrams
		self.allCaps = allCaps
		self.stats = stats
		self.unigramW = unigramW
		self.bigramW = bigramW
		self.unigramAndBigramW = unigramAndBigramW
		self.allCapsW = allCapsW
		self.statsW = statsW
		self.binary = binary
		self.testName = testName
		self.revs_text = self.loadData()
		self.X_train, self.X_test, self.y_train, self.y_test = self.splitTrainTestData()

	def loadData(self): # Get data
		print("Reading Data......")
		
		revs = pd.read_csv('rest_rev.csv', nrows=self.numRevs)
		# Text and stars only
		rev_text = revs[['stars', 'text']] #.reset_index().drop('business_id', axis=1)lf, numRevs):

		print("Data Read.")

		return rev_text

	def splitTrainTestData(self):
		X_train, X_test, y_train, y_test = train_test_split(self.revs_text['text'], self.revs_text['stars'], test_size=.3, random_state=119)
		return X_train, X_test, y_train, y_test

	def runModel(self, model, classifier, model_name):
		if self.pipeline == None:
			self.pipeline = self.buildPipeline(model, model_name)
		else:
			self.pipeline.steps.pop(1)
			self.pipeline.steps.append((model_name, model))

		self.pipeline.fit(self.X_train, self.y_train)

		pred = self.pipeline.predict(self.X_test)

		if classifier==True:
			self.evaluateClassifier(pred, self.y_test, model_name, self.pipeline)
		else:
			self.evaluateRegressor(pred, self.y_test, model_name, self.pipeline)

		print(model_name + " complete.\n")

	def evaluateClassifier(self, predicted, gold, classifier_type, pipeline):
		f = open("./" + self.testName + "/report.txt", 'a+')
		print(classifier_type + " evaluation:")
		f.write(classifier_type + " evaluation:\n")

		#convert to binary classification mapping
		if self.binary:
			for i in range(0, len(predicted)):
				if predicted[i] <= 3.0:
					predicted[i] = 0.0
				else:
					predicted[i] = 1.0

			gold = copy.deepcopy(gold)

			for i in gold.keys():
				if gold[i] <= 3.0:
					gold[i] = 0.0
				else:
					gold[i] = 1.0

		

		#print classification report
		print(metrics.classification_report(gold, predicted))
		f.write(metrics.classification_report(gold, predicted))

		#print model accuracy
		print("Accuracy: " + str(metrics.accuracy_score(gold, predicted)))
		f.write("Accuracy: " + str(metrics.accuracy_score(gold, predicted)) + "\n")


		#print confusion matrix
		print(metrics.confusion_matrix(gold, predicted))

		print()

		#generate confusion matrix table
		if self.binary:
			df_cm = pd.DataFrame(metrics.confusion_matrix(gold,predicted), range(2), range(2))
		else:
			df_cm = pd.DataFrame(metrics.confusion_matrix(gold,predicted), range(5), range(5))

		labels = [1,2,3,4,5]
		plt.title(classifier_type)
		sn.heatmap(df_cm, annot=True, cmap= sn.cm.rocket_r, xticklabels=labels, yticklabels=labels, fmt='g')
		plt.savefig("./" + self.testName + "/" + classifier_type + "_cm.png")
		plt.clf()

	def evaluateRegressor(self, predicted, gold, regressor_type, pipeline):
		f = open("./" + self.testName + "/report.txt", 'a+')

		print(regressor_type + " evaluation:")
		f.write(regressor_type + " evaluation:\n")

		print("Mean Squared Error: " + str(metrics.mean_squared_error(gold, predicted)))
		f.write("Mean Squared Error: " + str(metrics.mean_squared_error(gold, predicted)) + "\n")

		#map regression predictions to categorical classes
		rounded_pred = list(map(lambda x: round(x), predicted))
		for i in range(0, len(rounded_pred)):
			if rounded_pred[i] < 1.0:
				rounded_pred[i] = 1.0
			elif rounded_pred[i] > 5.0:
				rounded_pred[i] = 5.0

		#map classes to binary classes (3 stars and below is negative. 4 and 5 is positive)
		if self.binary:
			for i in range(0, len(predicted)):
				if predicted[i] <= 3.0:
					predicted[i] = 0.0
				else:
					predicted[i] = 1.0

			gold = copy.deepcopy(gold)

			for i in gold.keys():
				if gold[i] <= 3.0:
					gold[i] = 0.0
				else:
					gold[i] = 1.0


		#print classification report
		report = metrics.classification_report(gold, rounded_pred)
		print(report)
		f.write(report + "\n")

		#print mean squared error
		print("Mean Squared Error: " + str(metrics.mean_squared_error(gold, predicted)))
		f.write("Mean Squared Error: " + str(metrics.mean_squared_error(gold, predicted)) + "\n")

		#print model accuracy
		print("Accuracy: " + str(metrics.accuracy_score(gold, rounded_pred)))
		f.write("Accuracy: " + str(metrics.accuracy_score(gold, rounded_pred)) + "\n")

		#print confusion matrix
		cm = metrics.confusion_matrix(gold, rounded_pred)
		print(cm)

		#generate confusion matrix table
		if self.binary:
			df_cm = pd.DataFrame(cm, range(2), range(2))
		else:
			df_cm = pd.DataFrame(cm, range(5), range(5))
			
		labels = [1,2,3,4,5]
		plt.title(regressor_type)
		sn.heatmap(df_cm, annot=True, cmap= sn.cm.rocket_r, xticklabels=labels, yticklabels=labels, fmt='g')
		plt.savefig("./" + self.testName + "/" + regressor_type + "_cm.png")
		plt.clf()

	def buildPipeline(self, model, model_name):
		tList = []
		tWeights = {}

		if self.split:
			if self.minDF != 1:
				if not self.onlyBigrams:
					tList.append(('unigram', Pipeline([
						('tfidf', TfidfVectorizer(ngram_range=(1, 1), stop_words="english", min_df=self.minDF, token_pattern=r"(?u)\b\w[\w\']+\b")),
					])))

				if not self.onlyUnigrams:
					tList.append(('bigram', Pipeline([
						('tfidf', TfidfVectorizer(ngram_range=(2, 2), stop_words="english", min_df=self.minDF, token_pattern=r"(?u)\b\w[\w\']+\b")),
					])))
			else:
				if not self.onlyBigrams:
					tList.append(('unigram', Pipeline([
						('tfidf', TfidfVectorizer(ngram_range=(1, 1), stop_words="english", max_features=self.numUnigrams, token_pattern=r"(?u)\b\w[\w\']+\b")),
					])))

				if not self.onlyUnigrams:
					tList.append(('bigram', Pipeline([
						('tfidf', TfidfVectorizer(ngram_range=(2, 2), stop_words="english", min_df=self.numBigrams, token_pattern=r"(?u)\b\w[\w\']+\b")),
					])))
		else:
			if self.minDF != 1:
				tList.append(('unigram_and_bigram', Pipeline([
					('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=self.minDF, token_pattern=r"(?u)\b\w[\w\']+\b")),
				])))
			else:
				tList.append(('unigram_and_bigram', Pipeline([
					('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=self.numUnigrams, token_pattern=r"(?u)\b\w[\w\']+\b")),
				])))

		if self.allCaps:
			tList.append(('all_caps', Pipeline([
				('tfidf', CountVectorizer(ngram_range=(1, 1), stop_words=[x.upper() for x in stopwords.words('english')], min_df=75, token_pattern=r"(?u)\b[A-Z][A-Z]+\b", lowercase=False)),
			])))
			tWeights['all_caps'] = self.allCapsW

		if self.stats:
			tList.append(('rev_stats', Pipeline([
				('stats', TextStats()),
				('vect', DictVectorizer())
			])))

		tWeights['unigram_and_bigram'] = self.unigramAndBigramW
		tWeights['unigram'] = self.unigramW
		tWeights['bigram'] = self.bigramW
		tWeights['all_caps'] = self.allCapsW
		tWeights['stats'] = self.statsW

		
		pipeline = Pipeline([
			('union', FeatureUnion(
				transformer_list=tList,

				transformer_weights=tWeights,
			)),

			(model_name, model),
		])

		return pipeline

#modified from a scikit tutorial: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, reviews):
        return [{'length': len(text)
                 }
                for text in reviews]

if __name__ == "__main__":
	parser = OptionParser()

	parser.add_option("--nRevs", type="int", dest="numRevs", default=100000)
	parser.add_option("--nUnigrams", type="int", dest="numUnigrams", default=500)
	parser.add_option("--nBigrams", type="int", dest="numBigrams", default=100)
	parser.add_option("--minDF", type="int", dest="minDF", default=1)
	parser.add_option("--split", action="store_true", dest="splitBiAndUni", default=False)
	parser.add_option("--onlyUnigrams", action="store_true", dest="onlyUnigrams", default=False)
	parser.add_option("--onlyBigrams", action="store_true", dest="onlyBigrams", default=False)
	parser.add_option("--allCaps", action="store_true", dest="allCaps", default=False)
	parser.add_option("--stats", action="store_true", dest="stats", default=False)
	parser.add_option("--unigramW", type="float", dest="unigramW", default=1.0)
	parser.add_option("--bigramW", type="float", dest="bigramW", default=1.0)
	parser.add_option("--unigramAndBigramW", type="float", dest="unigramAndBigramW", default=1.0)
	parser.add_option("--allCapsW", type="float", dest="allCapsW", default=1.0)
	parser.add_option("--statsW", type="float", dest="statsW", default=1.0)
	parser.add_option("--binary", action="store_true", dest="binary", default=False)
	parser.add_option("--name", type="str", dest="testName", default="test")

	#parse command line
	(options, args) = parser.parse_args()

	if not os.path.exists(options.testName):
		os.makedirs(options.testName)

	#initialize engine
	engine = RestaurantEngine(options.numRevs, options.numUnigrams, options.numBigrams, options.minDF, options.splitBiAndUni, options.onlyUnigrams, options.onlyBigrams,
		options.allCaps, options.stats, options.unigramW, options.bigramW, options.unigramAndBigramW, options.allCapsW, options.statsW, options.binary, options.testName)

	#classification models
	engine.runModel(MultinomialNB(), True, "Naive Bayes")
	engine.runModel(LinearSVC(), True, "Linear SVC")
	engine.runModel(RandomForestClassifier(), True, "Random Forest")

	#regression models
	engine.runModel(linear_model.Ridge(alpha=.5), False, "Linear")
	engine.runModel(LinearSVR(), False, "Support Vector Regression")
	engine.runModel(DecisionTreeRegressor(), False, "Decision Tree Regression")