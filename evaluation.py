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

#returns accuracy of a confusion matrix
def accuracy(conf_matrix):
	correct = 0
	incorrect = 0
	for x in range(0,conf_matrix.shape[0]):
		for y in range(conf_matrix.shape[0]):
			if(x == y):
				correct += conf_matrix[x][y]
			else:
				incorrect += conf_matrix[x][y]
	return correct/(correct + incorrect)

def classification_rate(conf_matrix):
	pass

#outputs evaluation metrics to Evaluation_Output
def output_file():
	output_file = open('./Evaluation_Output', 'w')
	output_file.write('Iris Dataset:\n\n')
	output_file.write('Confusion Matrix\n')

	labels_iris = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	reader = csv.DictReader(open('./iris_out.csv', 'r'))
	iris_conf_matrix = confusion_matirx(reader, labels_iris)
	output_file.write(numpy.array_str(iris_conf_matrix))
	output_file.write('\n')

	acc = accuracy(iris_conf_matrix)
	#print(acc)
	output_file.write('accuracy = ' + str(acc) + '\n')
	output_file.write('error rate = '+ str(1-acc))

	output_file.write('\n\n')
	output_file.write('Income Dataset:\n\n')
	output_file.write('Confusion Matrix\n')

	reader = csv.DictReader(open('./income_out.csv', 'r'))
	labels_income = [' <=50K', ' >50K']
	income_conf_matrix = confusion_matirx(reader, labels_income)
	output_file.write(numpy.array_str(income_conf_matrix))
	output_file.write('\n')

	acc = accuracy(income_conf_matrix)
	output_file.write('accuracy = ' + str(acc) +'\n')
	output_file.write('error rate = '+ str(1-acc)+'\n')

	tp = income_conf_matrix[0][0]
	tn = income_conf_matrix[1][1]
	fn = income_conf_matrix[0][1]
	fp = income_conf_matrix[1][0]
	recall = tp / (tp + fp)
	precision = tp / (tp + fn)
	f_measure = (2*precision*recall)/(precision + recall)
	output_file.write('True Positive =' + str(tp)+'\n')
	output_file.write('True Negative ='+ str(tn)+'\n')
	output_file.write('False Negative =' + str(fn)+'\n')
	output_file.write('False Positive =' + str(fp)+'\n')
	output_file.write('Recall =' + str(recall)+'\n')
	output_file.write('Precision =' + str(precision)+'\n')

	#roc curve




output_file()
