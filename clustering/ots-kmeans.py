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
	cluster_SSE = {}
	for i in range(0,len(records)):
		record = records[i]
		prediction = predictions[i]
		centroid = centroids[prediction]
		distance = Euclidean(centroid, record)
		se = math.pow(distance, 2)
		SSE += se

		if prediction in cluster_SSE:
			cluster_SSE[prediction] += se
		else:
			cluster_SSE[prediction] = se
	print("Cluster SSEs:",cluster_SSE)
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
	kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='full')

	prediction = kmeans.fit_predict(records)

	score_SSE = abs(kmeans.score(records))

	calc_SSE = sse(records, prediction, kmeans.cluster_centers_)

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


	return score_SSE, calc_SSE, SSB

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
	score_SSE, SSE, SSB = kMeans(2, records)
	TSS = SSE + SSB
	for k in range(1, 20):
		print(k)
		score_kSSE, kSSE, kSSB = kMeans(k, records)
		kTSS = kSSE+kSSB

		if kTSS < TSS:
			k_ideal = k
			TSS = kTSS
			score_SSE = score_kSSE
			SSE = kSSE
			SSB = kSSB
	print("k value:", k_ideal)
	print("OTS SSE:", score_SSE, "SSE:", SSE, "SSB:", SSB)

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
	SS = kMeans(2, records)
	print("OTS SSE:", SS[0], "SSE:", SS[1], "SSB:", SS[2])


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
	score_SSE, SSE, SSB = kMeans(4, records)
	print("OTS SSE:", score_SSE, "SSE:", SSE, "SSB:", SSB)