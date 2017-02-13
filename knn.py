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

def kNNClassifier(kNearest, k):
	'''
	Determines the most popular class out of k records and calculates the posterior probability of that class.
	Requires len(kNearest) > 0
	'''
	class_voting = {}

	#For each of k nearest neighbors,
	for neighbor in kNearest:
		#If this neighbor's class has already been voted for by another neighbor,
		if neighbor[2] in class_voting:
			#Increment the number of votes
			class_voting[neighbor[2]] = class_voting[neighbor[2]]+1
		else:
			#Add class to dictionary with one vote
			class_voting[neighbor[2]] = 1
	
	#Determine "most popular" class
	predicted_class = max(class_voting, key=class_voting.get)

	#Calculate posterior probability
	posterior = class_voting[predicted_class]/k

	return [predicted_class, posterior]


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

	with open('Iris_Test.csv', "rt") as panda_iris_test_data:
		#Open test data with pandas library to provide ability to read entire columns at a time
		panda_iris_test = pandas.read_csv(panda_iris_test_data)

		#Open training data with CSV reader
		with open('Iris.csv', "rt") as iris_train_data:
			iris_train = csv.reader(iris_train_data)

			with open('Iris.csv', "rt") as panda_iris_train_data:
				#Open training data with pandas library to provide ability to read entire columns at a time
				panda_iris_train = pandas.read_csv(panda_iris_train_data)

				#Create (or overwrite) CSV file to hold output
				with open('iris_out.csv',"w") as output_file:
					output = csv.writer(output_file, dialect='excel')

					#Create header row for output CSV file
					output.writerow(['ID', 'Actual Class', 'Predicted Class', 'Posterior Probability'])

					#Create pandas dataframes for each column in test Iris data
					test_sepal_length = panda_iris_test.sepal_length
					test_sepal_width = panda_iris_test.sepal_width
					test_petal_length = panda_iris_test[' petal_length']
					test_petal_width = panda_iris_test[' petal_width']

					#pre-formatting test Iris data
					normalized_test = []
					for row in islice(iris_test, 1, None):
						#min-max normalization of each column
						normal_sl = (float(row[0]) - test_sepal_length.min().item())/(test_sepal_length.max().item()-test_sepal_length.min().item())
						normal_sw = (float(row[1]) - test_sepal_width.min().item())/(test_sepal_width.max().item()-test_sepal_width.min().item())
						normal_pl = (float(row[2]) - test_petal_length.min().item())/(test_petal_length.max().item()-test_petal_length.min().item())
						normal_pw = (float(row[3]) - test_petal_width.min().item())/(test_petal_width.max().item()-test_petal_width.min().item())
						#Save normalized rows with class
						normalized_test.append([normal_sl, normal_sw, normal_pl, normal_pw, row[4]])

					#Create pandas dataframes for each column in training Iris data
					train_sepal_length = panda_iris_test.sepal_length
					train_sepal_width = panda_iris_test.sepal_width
					train_petal_length = panda_iris_test[' petal_length']
					train_petal_width = panda_iris_test[' petal_width']

					#pre-formatting training Iris data
					normalized_train = []
					for row in islice(iris_train, 1, None):
						#min-max normalization of each column
						normal_sl = (float(row[0]) - train_sepal_length.min().item())/(train_sepal_length.max().item()-train_sepal_length.min().item())
						normal_sw = (float(row[1]) - train_sepal_width.min().item())/(train_sepal_width.max().item()-train_sepal_width.min().item())
						normal_pl = (float(row[2]) - train_petal_length.min().item())/(train_petal_length.max().item()-train_petal_length.min().item())
						normal_pw = (float(row[3]) - train_petal_width.min().item())/(train_petal_width.max().item()-train_petal_width.min().item())
						#Save normalized rows with class
						normalized_train.append([normal_sl, normal_sw, normal_pl, normal_pw, row[4]])


					#Initialize record ID (Iris data does not assign IDs)
					record_id = 1
					#Iterate over test records
					for record in normalized_test:
						kNearest = []

						#Initialize row ID
						row_id = 1
						#Compare test record to each row in training data set
						for row in normalized_train:
							if iris_measure == "E":
								#Calculate Euclidean distance between current test record and this row of training data
								proximity = Euclidean(record[0:4], row[0:4])
								#Add record ID, proximity, and class for training row as tuple to list
								kNearest.append((str(row_id), proximity, row[4]))
								#Sort list by proximity in ascending order (nearest first)
								kNearest = sorted(kNearest, key=lambda record:record[1])
							else:
								#Calculate Cosine similarity of record test record and this row of training data
								proximity = Cosine(record[0:4], row[0:4])
								#Add record ID, proximity, and class for training row as tuple to list
								kNearest.append((str(row_id), proximity, row[4]))
								#Sort list by proximity in descending order (most similar first)
								kNearest = sorted(kNearest, key=lambda record:record[1], reverse=True)
							#Ensure that list includes no more than k records
							del kNearest[k:]

							#Increment row ID
							row_id += 1

						#Add record ID and actual class to result row
						result = [record_id, record[4]]

						#Predict class of record from k-Nearest neighbors
						result += kNNClassifier(kNearest, k)
							
						#Write row to output file
						output.writerow(result)

						#Increment record ID
						record_id += 1

#Income Data
#Open test data with CSV reader
with open('income_te.csv', "rt") as income_test_data:
	income_test = csv.reader(income_test_data)

	with open('income_te.csv', "rt") as panda_income_test_data:
		#Open test data with pandas library to provide ability to read entire columns at a time
		panda_income_test = pandas.read_csv(panda_income_test_data)

		#Open training data with CSV reader
		with open('income_tr.csv', "rt") as income_train_data:
			income_train = csv.reader(income_train_data)

			with open('income_tr.csv', "rt") as panda_income_train_data:
				#Open training data with pandas library to provide ability to read entire columns at a time
				panda_income_train = pandas.read_csv(panda_income_train_data)

				#Create (or overwrite) CSV file to hold output
				with open('income_out.csv',"w") as output_file:
					output = csv.writer(output_file, dialect='excel')

					#Create header row for output CSV file
					output.writerow(['ID', 'Actual Class', 'Predicted Class', 'Posterior Probability'])


