import numpy as np
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import pickle

from flask import Flask, request, Response, json, abort
from flask import jsonify

############# FLASK
app = Flask(__name__)

############# PYTHON VALUE PASSING
V = np.array([]) # store INPUT
U = np.array([]) # store OUTPUT

def loadCSV(file, X, Y):
	data = np.loadtxt(open(file, "rb"), dtype = 'float, float, float, float, float, float, float, float, float, float', delimiter=",", skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9))
	global V,U
	t = 0
	for row in data:
			# if t < 50:
		temp = np.array([])
		for i in range(0,9):
			temp = np.append(temp, row[i])
		
		X = np.vstack((X,temp))
		Y = np.append(Y, row[9])
			# t += 1

	V = X
	U = Y

@app.route('/train', methods = ['GET'])
def train():
	X = np.array([0,1,2,3,4,5,6,7,8])
	Y = np.array([0])
	print("TRAINING")
	loadCSV('test3.csv', X, Y)

	clf = linear_model.BayesianRidge()
	clf.fit(V,U)
	# res = clf.predict([[0, 37.3084232, -121.8951796, 17108.0, 0.0, 13200.0, 0.0, 1900.0, 1327.0, 8.0, 4.0, 2.0, 95125.0]])
	# res = clf.predict([[37.3084232, -121.8951796, 0.0, 13200.0, 1900.0, 1327.0, 8.0, 4.0, 2.0, 713.0]])
	# res = clf.predict([[37.3084232, -121.8951796, 0.0, 13200.0, 1900.0, 1327.0, 8.0, 4.0, 2.0, 713.0, 2015]])
	# res = clf.predict([[37.3083563,-121.8948274,0,8500,1988,2371,7,3,2.5,713,2005]])
	res = clf.predict([[37.297955,-121.812944,0,1972,1799,4,2,95131,2017]])
	clf.
	print(res[0])

	###### SAVING MODEL
	pickle.dump(clf, open('trained_model.sav', 'wb'))
	loaded_model = pickle.load(open('trained_model.sav', 'rb'))
	print(loaded_model.predict([[37.297955,-121.812944,0,1972,1799,4,2,95131,2017]]))
	# print(loaded_model.predict([[37.3085298,-121.895274,0,12200,1947,1575,7,3,2,713,2011]]))
	return jsonify(result = res[0])

	################### SAVE/LOAD Model ################################
	# pickle.dump(clf, open('trained_model.sav', 'wb'))
	# loaded_model = pickle.load(open('trained_model.sav', 'rb'))
	# print(loaded_model.predict([[0, 37.3084232, -121.8951796,17108.0, 0.0, 13200.0, 0.0, 1900.0, 1327.0, 8.0, 4.0, 2.0, 95125.0]]))


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=1314, debug=True)
	