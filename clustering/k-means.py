
import csv 
import sys
import pandas
import random
import math 
import copy

#Euclidean distance between p1 and p2, ONLY USES ATTRIBUTRES PRESENT IN P1
def Euclidean(p1, p2):
	distance = 0

	for key in p1.keys():
		val1 = p1[key]
		val2 = p2[key]
		distance += abs(val1-val2)**2
	distance = math.sqrt(distance)
	
	return distance

def kmeans(k, reader):
	#create initialization points

	#column_header->(min, max) of each attribute 
	limits_and_means = find_attribute_limits_and_mean(reader);

	attribute_limits = limits_and_means[0]
	#Get means of all attributes and convert to single dictionary
	attribute_means = limits_and_means[1][0]
	for i in range (1, len(limits_and_means[1])):
		attribute_means.update(limits_and_means[1][i])
	#print("means:", attribute_means)

	centroids = [] #list of dictionaries, [{attribute_name: value}]
	for i in range(0, k):
		point = {}
		for header, limits in attribute_limits.items():
			point[header] = random.uniform(limits[0], limits[1]) 

		centroids.append(point)

	times_unchanged = 0
	attributes = centroids[0].keys();

	reader['guessed_cluster'] = -1

	while (times_unchanged <2): 
		for row in reader.iterrows(): 
			point = row[1] #dictionary of attributes to values, includes extra
			id = row[0]
			min_distance = 100
			cluster = 0;
			for centroid in centroids:
				distance = Euclidean(centroid, point)
				if(distance<min_distance):
					min_distance = distance;
					reader.set_value(id,'guessed_cluster', cluster)
				cluster += 1;

		previous_centroids = copy.deepcopy(centroids)
		#print('centroids ', centroids)
		#make centroids zero
		for centroid in centroids:
			for attribute in attributes:
				centroid[attribute] = 0 

		cluster_totals = []
		for i in range(0,k):
			cluster_totals.append(0)

		for row in reader.iterrows():
			point = row[1]
			for i in range (0,k):
				if point['guessed_cluster'] == i:
					cluster_totals[i] += 1
					for attribute in attributes:
						centroids[i][attribute] += point[attribute]

		for i in range(0,k):
			centroid = centroids[i]
			for attribute in attributes:
				centroid[attribute] /= cluster_totals[i]

		unchanged = True

		for centroid, previous_centroid in zip(centroids, previous_centroids):
			if centroid != previous_centroid:
				unchanged = False
				if times_unchanged!= 0:
					times_unchanged = 0

		if(unchanged):
			times_unchanged+=1

	cluster_column = reader['guessed_cluster']	
	cluster_totals = []
	for i in range(0,k):
		cluster_totals.append(0)
		for cluster in cluster_column:
			if cluster == i:
				cluster_totals[i] += 1


	with open('./output_hard_norm2.csv', 'w') as output_file:
		reader.to_csv(output_file)

	#centroids= list of dicts [{attibute:value}], attribute means= dict {attribute: mean}, cluster_totals = list of totals, corresponding with the index of the centroid
	return (centroids, attribute_means, cluster_totals)


def find_attribute_limits_and_mean(reader):
	#dictionary of attribute ids -> (max, min) of the attribute values
	limits = {}

	#list of dictionaries [{attribute:mean}]
	means = []
	for header in reader.columns.values:
		if header != 'ID' and header!= 'class' and header != 'cluster' and header!= 'class' and header != 'quality':
			attribute_values = reader[header].tolist()
			limits[header] = (min(attribute_values), max(attribute_values))
			means.append({header: (sum(attribute_values)/float(len(attribute_values)))})

	return (limits, means)

def sse(reader, centroids):
	SSE = 0
	cluster_SSE = {}
	for row in reader.iterrows():
		point = row[1]
		guessed_cluster = int(point['guessed_cluster'])
		centroid = centroids[guessed_cluster]
		distance = Euclidean(centroid, point)
		se = math.pow(distance, 2)
		SSE += se

		if guessed_cluster in cluster_SSE:
			cluster_SSE[guessed_cluster] += se
		else:
			cluster_SSE[guessed_cluster] = se
	print("Cluster SSEs:", cluster_SSE)
	return SSE

def ssb(centroids, cluster_totals, attribute_means):
	SSB = 0
	for i in range(0, len(centroids)):
		cluster_size = cluster_totals[i]
		centroid = centroids[i]
		distance = Euclidean(centroid, attribute_means)
		SSB += cluster_size*(math.pow(distance,2))
	return SSB


#expects arguments of k, then 1 for easy, 2 for hard, and 3 for wine
def main():
	file_name = ''
	if(sys.argv[2] == '1'):
		file_name = './TwoDimEasyNormalized.csv'
	elif(sys.argv[2] == '2'):
		file_name = './TwoDimHardNormalized.csv'
	elif(sys.argv[2] == '3'):
		file_name = './wine_normalized.csv'
	else:
		print('invalid arguments\n')
		return

	with open(file_name) as data:
		reader = pandas.read_csv(data)

	centroids, attribute_means, cluster_totals = kmeans(int(sys.argv[1]), reader)

	SSE = sse(reader, centroids)
	print("SSE:", SSE)
	SSB = ssb(centroids, cluster_totals, attribute_means)
	print("SSB:", SSB)
	TSS = SSE + SSB

	if(sys.argv[2] == '3'):
		#If wine data set, try a number of different settings for k
		k_ideal = sys.argv[1]
		for k in range(1, 7):
			print("Testing k value:", k)
			if k != sys.argv[1]:
				centroids,attribute_means, cluster_totals = kmeans(k, reader)
				kSSE = sse(reader, centroids)
				kSSB = ssb(centroids, cluster_totals, attribute_means)
				kTSS = kSSE + kSSB

				if kTSS < TSS and kSSB != 0:
					k_ideal = k
					SSE = kSSE
					SSB = kSSB
					TSS = kTSS
		print("Optimal k value for wine:", k_ideal)
		print("SSE:", SSE)
		print("SSB:", SSB)


if __name__ == "__main__":main()
