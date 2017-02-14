import csv
import knn
import numpy

#returns confusion matrix
#params: cvs reader for input file, array of class labels
def confusion_matirx(reader, labels):

	actual = []
	predicted = []
	for row in reader:
		actual.append(row['Actual Class'])
		predicted.append(row['Predicted Class'])

	conversion = {}
	x = 0
	for label in labels:
		conversion[label] = x
		x +=1

	actual = numpy.array(actual)
	predicted = numpy.array(predicted)
	conf_matrix = numpy.zeros((len(labels), len(labels)))
	for actual_class, predicted_class in zip(actual, predicted):
		#print(actual_class, predicted_class)
		conf_matrix[conversion[actual_class]][conversion[predicted_class]] +=1
		#print(conf_matrix)
	return conf_matrix

def income_eval(conf_matrix):
	#TP, FP, TN, FN, precision, f-measure, roc curve
	pass

def error_rate(conf_matrix):
	pass
	#error_rate = (fn+fp)/(tp + fn + fp + tn)
def classification_rate(conf_matrix):
	pass

def output_file():
	#confusion matirx, classification and error rates for test set for a few k values
	#'ID', 'Actual Class', 'Predicted Class', 'Posterior Probability']
	output_file = open('./Evaluation_Output', 'w')
	output_file.write('Iris Dataset:\n\n')
	output_file.write('Confusion Matrix\n')

	labels_iris = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	reader = csv.DictReader(open('./iris_out.csv', 'r'))
	iris_conf_matrix = confusion_matirx(reader, labels_iris)
	output_file.write(numpy.array_str(iris_conf_matrix))

	output_file.write('\n\n')
	output_file.write('Income Dataset:\n\n')
	output_file.write('Confusion Matrix\n')

	reader = csv.DictReader(open('./income_out.csv', 'r'))
	labels_income = [' <=50K', ' >50K']
	income_conf_matrix = confusion_matirx(reader, labels_income)
	output_file.write(numpy.array_str(income_conf_matrix))


	#output_file.write('error rate = ' + error_rate + '\n\n')


output_file()
