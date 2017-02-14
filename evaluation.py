import csv
import knn
import numpy


#'ID', 'Actual Class', 'Predicted Class', 'Posterior Probability']

#produces confusion matrix, classificatoin, and errors rates
def confusion_matirx(reader, output_file, labels):

	#confusion matirx, classification and error rates for test set for a few k values
	actual = []
	predicted = []
	for row in reader:
		actual.add(row['Actual Class'])
		predicted.add(row['Predicted Class']

	actual = numpy.array(actual)
	predicted = numpy.array(predicted)
	conf_matrix = numpy.zeros(len(labels), len(labels))
	for actual_class, predicted_class in zip(actual, predicted):
		conf_matrix[actual]

	output_file.write('Confusion Matrix\n\n')


	error_rate = (fn+fp)/(tp + fn + fp + tn)
	#classification rate???
	output_file.write('error rate = ' + error_rate + '\n\n')
	output_file.write('Confusion Matrix = \n')


	input_file.close()

def income_eval():
	#TP, FP, TN, FN, precision, f-measure, roc curve

def main():
	labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

	for k in range(5, 9):
		pass 
		#call knn
		#call evaluation income
		#call evaluation iris



