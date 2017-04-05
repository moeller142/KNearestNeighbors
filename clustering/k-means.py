
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
	attribute_means = limits_and_means[1]
	print(attribute_means)

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
		print('centroids ', centroids)
		#make centroids zero
		for centroid in centroids:
			for attribute in attributes:
				centroid[attribute] = 0 

		for row in reader.iterrows():
			point = row[1]
			for i in range (0,k):
				if point['guessed_cluster'] == i:
					for attribute in attributes:
						centroids[i][attribute] += point[attribute]

		for centroid in centroids:
			for attribute in attributes:
				centroid[attribute] /= reader.shape[0]

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


	with open('./wine_output.csv', 'w') as output_file:
		reader.to_csv(output_file)

	#centroids= list of dicts [{attibute:value}], attribute means= list of dicts [{attribute, mean}], cluster_totals = list of totals, corresponding with the index of the centroid
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
	for row in reader.iterrows():
		point = row[1]
		centroid = centroids[int(point['guessed_cluster'])]
		distance = Euclidean(centroid, point)
		SSE += math.pow(distance, 2)
	return SSE

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

	centroids,attribute_means, cluster_totals = kmeans(int(sys.argv[1]), reader)

	SSE = sse(reader, centroids)
	print("SSE:", SSE)


if __name__ == "__main__":main()
