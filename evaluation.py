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
		conf_matrix[conversion[actual_class]][conversion[predicted_class]] +=1
	return conf_matrix

def income_eval(conf_matrix):
	#TP, FP, TN, FN, precision, f-measure, roc curve
	pass

def error_rate(conf_matrix):
	#error_rate = (fn+fp)/(tp + fn + fp + tn)
	pass

def classification_rate(conf_matrix):
	pass

def main():
	#confusion matirx, classification and error rates for test set for a few k values
	#'ID', 'Actual Class', 'Predicted Class', 'Posterior Probability']
	output_file = open('./Evaluation_Output', 'w')
	output_file.write('Iris Dataset:\n\n')
	output_file.write('Confusion Matrix\n')

	labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	reader = csv.DictReader(open('./iris_out.csv', 'r'))
	iris_conf_matrix = confusion_matirx(reader, labels)
	print(iris_conf_matrix)
	#output_file.write(iris_conf_matrix)


	#output_file.write('error rate = ' + error_rate + '\n\n')

if __name__ == '__main__':
	main()


