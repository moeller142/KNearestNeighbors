
import csv 
import sys
import pandas
import random
import math 

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
	attribute_limits = find_attribute_limits(reader);

	centroids = [] #list of dictionaries, [{attribute_name: value}]
	for i in range(0, k):
		point = {}
		for header, limits in attribute_limits.items():
			point[header] = random.uniform(limits[0], limits[1]) 

		centroids.append(point)

	times_unchanged = 0
	print(centroids)
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
			print('guessed cluster', reader.ix[id]['guessed_cluster'])

		previous_centroids = centroids
		print('previous centroids ', previous_centroids)
		print('centroids ', centroids)
		#make centroids zero
		for centroid in centroids:
			for attribute in attributes:
				centroid[attribute] = 0 

		print('centroids 2 ', centroids)

		for row in reader.iterrows():
			point = row[1]
			for i in range (0,k):
				if point['guessed_cluster'] == i:
					for attribute in attributes:
						centroids[i][attribute] += point[attribute]

		print('centroids 3 ', centroids)

		for centroid in centroids:
			for attribute in attributes:
				centroid[attribute] /= reader.shape[0]

		print('centroids 4 ', centroids)
		print('previous centroids 2', previous_centroids)


		unchanged = True

		for centroid, previous_centroid in zip(centroids, previous_centroids):
			if centroid != previous_centroid:
				unchanged = False

		if(unchanged):
			print('unchanged');
			times_unchanged+=1


def find_attribute_limits(reader):
	#dictionary of attribute ids -> (max, min) of the attribute values
	limits = {}

	for header in reader.columns.values:
		if header != 'ID' and header!= 'class' and header != 'cluster':
			attribute_values = reader[header].tolist()
			limits[header] = (min(attribute_values), max(attribute_values))

	return limits

#expects arguments of k, then 1 for easy, 2 for hard, and 3 for wine
def main():
	file_name = ''
	if(sys.argv[2] == '1'):
		file_name = './TwoDimEasy.csv'
	elif(sys.argv[2] == '2'):
		file_name = './TwoDimHard.csv'
	elif(sys.argv[2] == '3'):
		file_name = './wine.csv'
	else:
		print('invalid arguments\n')
		return

	with open(file_name) as data:
		reader = pandas.read_csv(data)

	kmeans(int(sys.argv[1]), reader)

if __name__ == "__main__":main()