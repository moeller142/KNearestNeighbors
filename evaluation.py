import csv
import numpy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import pandas as pd
from ggplot import *

#returns confusion matrix
#params: cvs reader for input file, array of class labels
def confusion_matirx(reader, labels):

	actual = []
	predicted = []

	a = []
	p = []

	conversion = {' <=50K':0, ' >50K':1}
	for row in reader:
		actual.append(row['Actual Class'])
		predicted.append(row['Predicted Class'])

		a.append(conversion[row['Actual Class']])
		p.append(conversion[row['Predicted Class']])


	roc_curve(a, p)

	conversion = {}
	x = 0
	for label in labels:
		conversion[label] = x
		x +=1

	print(conversion)			
	actual = numpy.array(actual)
	predicted = numpy.array(predicted)
	conf_matrix = numpy.zeros((len(labels), len(labels)))
	for actual_class, predicted_class in zip(actual, predicted):
		conf_matrix[conversion[actual_class]][conversion[predicted_class]] +=1
	print(conf_matrix)
	return conf_matrix

def roc_curve(actual, predicted):
	false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(actual, predicted)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	plt.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()

	# fpr, tpr, _ = metrics.roc_curve(actual, predicted, pos_label = ' <=50K')

	# df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))	
	# p = ggplot(df, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')
	# auc = metrics.auc(fpr,tpr)
	# g = ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +geom_area(alpha=0.2) +geom_line(aes(y='tpr')) +ggtitle("ROC Curve w/ AUC=%s" % str(auc))
	# print(p)
	# print(g)
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



#outputs evaluation metrics to Evaluation_Output
def output_file():
	output_file = open('./Evaluation_Output', 'w')
	output_file.write('Iris Dataset:\n\n')
	output_file.write('Confusion Matrix\n')

	# labels_iris = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	# reader = csv.DictReader(open('./iris_out.csv', 'r'))
	# iris_conf_matrix = confusion_matirx(reader, labels_iris)
	# output_file.write(numpy.array_str(iris_conf_matrix))
	# output_file.write('\n')

	# acc = accuracy(iris_conf_matrix)
	# #print(acc)
	# output_file.write('accuracy = ' + str(acc) + '\n')
	# output_file.write('error rate = '+ str(1-acc))

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
