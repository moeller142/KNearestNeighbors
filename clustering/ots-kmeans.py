from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import csv
import numpy
import math

def Euclidean(record1, record2):
	'''
	Calculates Euclidean distance between two records with an arbitrary, identical number of dimensions
	'''
	distance = 0
	for i in range(0,len(record1)):
			distance += math.pow(abs(record1[i]-record2[i]),2)
	distance = math.sqrt(distance)
	return distance

def sse(records, predictions, centroids):
	SSE = 0
	for i in range(0,len(records)):
		record = records[i]
		prediction = predictions[i]
		centroid = centroids[prediction]
		distance = Euclidean(centroid, record)
		SSE += math.pow(distance, 2)
	return SSE

def kMeans(k, records, classes):
	#TODO parameters? Do we tell it how many clusters because we know the data set?
	kmeans = KMeans(n_clusters=k)

	#prediction = cross_val_predict(kmeans, records, classes)
	prediction = kmeans.fit_predict(records, classes)
	ars = metrics.adjusted_rand_score(classes, prediction)
	print("Adjusted Rand Index:", ars)
	ami = metrics.adjusted_mutual_info_score(classes, prediction)
	print("Adjusted Mutual Information Score:", ami)
	homogeneity = metrics.homogeneity_score(classes, prediction)
	print("Homogeneity Score:", homogeneity)
	completeness = metrics.completeness_score(classes, prediction)
	print("Completeness Score:", completeness)
	vMeasure = metrics.v_measure_score(classes, prediction)
	print("V-measure:", vMeasure)
	#TODO: other performance measures??

	SSE = sse(records, prediction, kmeans.cluster_centers_)

	return SSE

#Open CSV file containing data set
with open('wine.csv', "rt") as wine_data:
	wine = csv.reader(wine_data)
	wine = list(wine)

	records = []
	classes = []
	for record in wine[1:]:
		records.append(numpy.array(record[1:12]).astype(numpy.float))
		classes.append(int(record[12]))

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Wine dataset")
	SSE = kMeans(2, records, classes)
	print("SSE:", SSE)

#Open CSV file containing data set
with open('TwoDimEasy.csv', "rt") as easy_data:
	easy = csv.reader(easy_data)
	easy = list(easy)

	records = []
	classes = []
	for record in easy[1:]:
		records.append(numpy.array(record[1:3]).astype(numpy.float))
		classes.append(record[3])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Easy dataset")
	SSE = kMeans(2, records, classes)
	print("SSE:", SSE)

#Open CSV file containing data set
with open('TwoDimHard.csv', "rt") as hard_data:
	hard = csv.reader(hard_data)
	hard = list(hard)

	records = []
	classes = []
	for record in hard[1:]:
		records.append(numpy.array(record[1:3]).astype(numpy.float))
		classes.append(record[3])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Hard dataset")
	SSE = kMeans(4, records, classes)
	print("SSE:", SSE)