File name: proximity.py
Author: Jennifer Barry

Dependencies:
	This program expects python3 to be installed.
	This program requires the "NumPy" library for Python.
		To install, use the command "sudo apt-get install python-pip"
	This program requires the "pandas" library for Python.
		To install, use the command "sudo apt-get install python3-pandas"
Input Data:
	Input data files should be placed in the same directory as the program file.

Execution:
	Once dependencies have been resolved, the program can be run from a linux command line with the following command:
		python3 proximity.py <optional value of k> <optional indicator for preferred measurement> <secondary indicator>
	
		<optional value of k> is expected to be an integer value specifying the number of "nearest neighbors" to include in the output.

		<optional indicator for preferred measurement> is expected to be either:
			- The letter E (capitalized) for Euclidean distance
			- The letter C (capitalized) for Cosine similarity
			This argument specifies the preferred proximity measurement to be applied.
		
		<secondary indicator> follows the same rules as <optional indicator for preferred measurement> above, except it will only be applied to the Income data set.
			The value chosen for the first indicator will only be used on the Iris data set.
		
		
		NOTE: While all bracketed arguments listed above are optional, they must be in the order listed if they are included.
			For example, the program will fail if only a secondary indicator is provided without including the k value and first indicator.
			Examples of proper input include:
				- python3 proximity.py
				- python3 proximity.py 3
				- python3 proximity.py 10 C
				- python3 proximity.py 6 E C
Behavior:
	The program takes as input two data sets: the Iris data set (Iris.csv) and the Income data set (income_tr.csv).
	For each data set, individually, the proximity of each record to all other records in the data set is calculated.
	For each record, the "k" nearest records are stored (where k is a positive integer).
	Upon completion, the program outputs a new data set containing each record's ID, and the IDs and proximities of the "k" records which are closest to it. These records are displayed in order from nearest to farthest. 

	The program accepts a number of parameters.
		- The value for k can be changed by the user. By default, it is 5.
		- The proximity measure used to calculate how near a given record is to another can be chosen by the user.
			- There are two options: Euclidean distance (a measure of dissimilarity), and Cosine similarity.
			- By default, the program will calculate Euclidean distance in determining proximity.
		- The proximity measure for the second data set (the Income data set) can be chosen independently of the first data set.
			- By default, the second data set will use the same proximity measure as the first data set.
Output Data:
	Output data files will be stored in CSV format in the same directory as the program file.
	The output files will be named "iris_out.csv" and "income_out.csv" for the Iris and Income data sets, respectively.