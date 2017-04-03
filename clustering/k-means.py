
import csv 
import sys
import pandas
import random

def Euclidean(record, row):
	'''
	Calculates Euclidean distance between two records with an arbitrary, identical number of dimensions
	'''
	distance = 0
	for i in range(0,len(record)):
			distance += abs(record[i]-row[i])**2
	distance = math.sqrt(distance)
	return distance

def kmeans(k, reader):
	#create initialization points

	#column_header->(min, max) of each attribute 
	attribute_limits = find_attribute_limits(reader);
	print(attribute_limits)

	#list of centroids, k-tuples 
	print(k)
	centroids = []
	for i in range(0, k):
		point = {}
		for header, limits in attribute_limits.items():
			point[header] = random.uniform(limits[0], limits[1]) 

		centroids.append(point)

	previous_centroids = centroids
	print(centroids)
	print(attribute_limits)
	times_unchanged = 0
	
	'''while (times_unchanged <3){

	}'''

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

	print(sys.argv[1])
	print(int(sys.argv[1]))
	kmeans(int(sys.argv[1]), reader)

if __name__ == "__main__":main()