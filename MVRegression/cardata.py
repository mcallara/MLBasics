import csv
import numpy as np

class cardata(object):
	#["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]
	@staticmethod
	def lineparser(line):
		'''labels: mpg y el resto data'''
		label=line[0]
		data=line[1:-1]
		[ele=='?' for ele in data]
		return label,data

	@staticmethod
	def list2numpy(listob):
		array=np.array(listob)
		array[array=='?']=0
		return np.float64(array)

	def __init__(self,filename):
		self.labels=[]
		self.data=[]
		self.header=["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]
		with open(filename) as f:
			reader = csv.reader(f,delimiter=';')
			for line in reader:
				label,data=self.lineparser(line)
				self.labels.append(label)
				self.data.append(data)
		self.ndata=self.list2numpy(self.data)
		self.nlabels=self.list2numpy(self.labels)


filename='auto-mpg.data'
cars=cardata(filename)