from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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

def ssb(clusters, centroids, overall_centroid):
	SSB = 0
	for i in range(0, len(centroids)):
		size = clusters[i]
		centroid = centroids[i]
		distance = Euclidean(centroid, overall_centroid)
		SSB += size*(math.pow(distance,2))
	return SSB

def kMeans(k, records):
	#TODO parameters? Do we tell it how many clusters because we know the data set?
	kmeans = KMeans(n_clusters=k)

	prediction = kmeans.fit_predict(records)
	'''
	ars = metrics.adjusted_rand_score(clusters, prediction)
	print("Adjusted Rand Index:", ars)
	ami = metrics.adjusted_mutual_info_score(clusters, prediction)
	print("Adjusted Mutual Information Score:", ami)
	homogeneity = metrics.homogeneity_score(clusters, prediction)
	print("Homogeneity Score:", homogeneity)
	completeness = metrics.completeness_score(clusters, prediction)
	print("Completeness Score:", completeness)
	vMeasure = metrics.v_measure_score(clusters, prediction)
	print("V-measure:", vMeasure)
	#TODO: other performance measures??
	'''

	#NOTE! Abs value of this = SSE
	#print(kmeans.score(records))

	SSE = sse(records, prediction, kmeans.cluster_centers_)

	#SSB
	k1 = KMeans(n_clusters=1)
	k1.fit_predict(records)
	overall_centroid = k1.cluster_centers_[0]

	clusters = {}
	for num in prediction:
		if num in clusters:
			clusters[num] += 1
		else:
			clusters[num] = 1

	SSB = ssb(clusters, kmeans.cluster_centers_, overall_centroid)


	return [SSE, SSB]

def dbSCAN(records, clusters):
	dbscan = DBSCAN()

	prediction = dbscan.fit_predict(records, clusters)

	print(dbscan.core_sample_indices_)
	SSE = sse(records, prediction, dbscan.core_sample_indices_)

	return SSE


#Open CSV file containing data set
with open('wine.csv', "rt") as wine_data:
	wine = csv.reader(wine_data)
	wine = list(wine)

	records = []
	clusters = []
	for record in wine[1:]:
		records.append(numpy.array(record[1:12]).astype(numpy.float))
		clusters.append(int(record[12]))

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Wine dataset")
	

	k_ideal = 2
	SS = kMeans(2, records)
	SSE = SS[0]
	SSB = SS[1]
	TSS = SSE + SSB
	for k in range(3, 20):
		print(k)
		kSS = kMeans(k, records)
		kTSS = kSS[0]+kSS[1]

		if kTSS < TSS:
			k_ideal = k
			TSS = kTSS
			SSE = kSS[0]
			SSB = kSS[1]
	print("k value:", k_ideal)
	print("k-means SSE:", kSS[0], "SSB:", kSS[1])

	#dSS = dbSCAN(records, clusters)
	#print("DBSCAN SSE:", dSS)

#Open CSV file containing data set
with open('TwoDimEasy.csv', "rt") as easy_data:
	easy = csv.reader(easy_data)
	easy = list(easy)

	records = []
	clusters = []
	for record in easy[1:]:
		records.append(numpy.array(record[1:3]).astype(numpy.float))
		clusters.append(record[3])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Easy dataset")
	kSS = kMeans(2, records)
	print("k-means SSE:", kSS[0], "SSB:", kSS[1])

	#dSS = dbSCAN(records, clusters)
	#print("DBSCAN SSE:", dSS)


#Open CSV file containing data set
with open('TwoDimHard.csv', "rt") as hard_data:
	hard = csv.reader(hard_data)
	hard = list(hard)

	records = []
	clusters = []
	for record in hard[1:]:
		records.append(numpy.array(record[1:3]).astype(numpy.float))
		clusters.append(record[3])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Hard dataset")
	kSS = kMeans(4, records)
	print("k-means SSE:", kSS[0], "SSB:", kSS[1])

	#dSS = dbSCAN(records, clusters)
	#print("DBSCAN SSE:", dSS)