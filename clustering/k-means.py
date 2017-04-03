
import csv 
import sys
import pandas
import random

def Euclidean(p1, p2):
	distance = 0

	for key in p1.keys()
		val1 = p1[key]
		val2 = p2[key]
		distance += abs(val1-val2)**2
	distance = math.sqrt(distance)
	
	return distance

def kmeans(k, reader):
	#create initialization points

	#column_header->(min, max) of each attribute 
	attribute_limits = find_attribute_limits(reader);

	#list of centroids, k-tuples 
	print(k)
	centroids = []
	for i in range(0, k):
		point = {}
		for header, limits in attribute_limits.items():
			point[header] = random.uniform(limits[0], limits[1]) 

		centroids.append(point)

	previous_centroids = centroids
	times_unchanged = 0

	while (times_unchanged <3):
		for row in reader.iterrows():
			point = row[1]
			min_distance = 100
			cluster = 0;
			for centroid in centroids:
				distance = Euclidean(point, centroid)
				if(distance<min_distance):
					min_distance = distance;
					point['guessed_cluster'] = cluster
				cluster += 1;

		for row in reader.iterrows():


	

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
	print(sys.argv)
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