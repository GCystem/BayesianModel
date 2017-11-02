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
	data = np.loadtxt(open(file, "rb"), dtype = 'float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float', delimiter=",", skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
	global V,U
	t = 0
	for row in data:
			# if t < 50:
		temp = np.array([])
		for i in range(0,13):
			temp = np.append(temp, row[i])
		
		X = np.vstack((X,temp))
		Y = np.append(Y, row[15])
			# t += 1

	V = X
	U = Y

	# with open(file, 'rb') as csvfile:
	# 	reader = csv.DictReader(csvfile)
	# 	t = 0
	# 	for row in reader:
	# 		if t < 50:
	# 			# print(row['Value at Median $ per Sq. Ft.(2017)'])
	# 			T = []
	# 			T.append(float(row['Lat']))
	# 			T.append(float(row['Lng']))
	# 			T.append(float(row['Tax rate area']))
	# 			# T.append(float(row['Neighborhood']))
	# 			# T.append(float(row['Lot sq. ft.']))
	# 			# T.append(float(row['Property class']))
	# 			# T.append(float(row['Year built']))
	# 			# T.append(float(row['Square feet']))
	# 			# T.append(float(row['Rooms']))
	# 			T.append(float(row['Bedrooms']))
	# 			T.append(float(row['Bathrooms']))
	# 			T.append(float(row['ZIP code']))
	# 			print(T)
	# 			T = np.array(X)
	# 			X.append(T)
	# 			Y.append(float(row['Value at Median $ per Sq. Ft.(2017)']))
	# 		t += 1

@app.route('/train', methods = ['GET'])
def train():
	X = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
	Y = np.array([0])
	print("TRAINING")
	loadCSV('test.csv', X, Y)

	# print(V)
	# print(U)

	clf = linear_model.BayesianRidge()
	clf.fit(V,U)
	res = clf.predict([[0, 37.3084232, -121.8951796, 17108.0, 0.0, 13200.0, 0.0, 1900.0, 1327.0, 8.0, 4.0, 2.0, 95125.0]])

	print(res[0])
	return jsonify(result = res[0])

	################### SAVE/LOAD Model ################################
	# pickle.dump(clf, open('trained_model.sav', 'wb'))
	# loaded_model = pickle.load(open('trained_model.sav', 'rb'))
	# print(loaded_model.predict([[0, 37.3084232, -121.8951796,17108.0, 0.0, 13200.0, 0.0, 1900.0, 1327.0, 8.0, 4.0, 2.0, 95125.0]]))

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=1314, debug=True)
	