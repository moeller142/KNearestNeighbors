import csv

def evaluation(file):
	input_file = open(file, "w")
	reader = csv.DictReader(input_file)
	