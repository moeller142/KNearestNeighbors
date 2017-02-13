import sys
import csv
import pandas
import numpy
import math
from itertools import islice

#Proximity Functions
#NOTE: Euclidean distance is a measure of dissimilarity
def Euclidean(record, row):
	'''
	Calculates Euclidean distance between two records with an arbitrary, identical number of dimensions
	'''
	distance = 0
	for i in range(0,len(record)):
			distance += abs(record[i]-row[i])**2
	distance = math.sqrt(distance)
	return distance

#NOTE: Cosine similarity is a measure of similarity
def Cosine(record, row):
	'''
	Calculates Cosine similarity of two records with an arbitrary number of dimensions
	'''
	#Initialize values
	dot_product = 0
	record_length = 0
	row_length = 0

	for i in range(0,len(record)):
		record_length += record[i]**2
		row_length += row[i]**2
		dot_product += record[i]*row[i]

	#take the square root of the sum of squares
	record_length = math.sqrt(record_length)
	row_length = math.sqrt(row_length)

	#return cosine similarity
	return dot_product/(record_length*row_length)

def Binary(element1, element2, similarity):
	if element1 == element2:
		if similarity:
			return 1
		else:
			return 0
	else:
		if similarity:
			return 0
		else:
			return 1

#Initialization
#initialize k to a default value of 5
k = 5

#If user has provided a value for k, replace the initial value
if len(sys.argv) > 1:
	k = int(sys.argv[1])

#By default, this program will calculate a Euclidean distance as the proximity measure
iris_measure = "E"
#If user has provided an argument specifying the type of proximity measure to use, save that preference
if len(sys.argv) > 2:
	#If user enters "C", change measure to Cosine Similarity
	if sys.argv[2] == "C":
		iris_measure = "C"
	#If user has entered anything other than the two options ("E" and "C"), output a warning message to console.
	elif sys.argv[2]!="E":
		print("Warning: user has entered invalid argument for proximity measure. The default measure (Euclidean distance) will be used.")

#By default, both data sets will use the same proximity measure
income_measure = iris_measure
#If user has provided a secondary measure to use,
if len(sys.argv) > 3:
	#record user preference
	if sys.argv[3] == "E":
		income_measure = "E"
	elif sys.argv[3] == "C":
		income_measure = "C"
	else:
		#If user has entered anything other than the two options ("E" and "C"), output a warning message to console.
		if iris_measure == "E":
			print("Warning: user has entered invalid argument for secondary proximity measure. By default, the primary measure (Euclidean distance) will be used.")
		else:
			print("Warning: user has entered invalid argument for secondary proximity measure. By default, the primary measure (Cosine similarity) will be used.")

#Iris Data
#Open test data with CSV reader
with open('Iris_Test.csv', "rt") as iris_test_data:
	iris_test = csv.reader(iris_test_data)

	with open('Iris.csv', "rt") as panda_iris_test_data:
		#Open test data with pandas library to provide ability to read entire columns at a time
		panda_iris_test = pandas.read_csv(panda_iris_test_data)

		#Open training data with CSV reader
		with open('Iris_Test.csv', "rt") as iris_train_data:
			iris_train = csv.reader(iris_test_data)

			with open('Iris.csv', "rt") as panda_iris_train_data:
				#Open training data with pandas library to provide ability to read entire columns at a time
				panda_iris_train = pandas.read_csv(panda_iris_train_data)

				#Create (or overwrite) CSV file to hold output
				with open('iris_out.csv',"w") as output_file:
					output = csv.writer(output_file, dialect='excel')

					#Create header row for output CSV file
					output.writerow(['ID', 'Actual Class', 'Predicted Class', 'Posterior Probability'])