from numpy import exp, array, random, dot
import numpy
import matplotlib.pyplot as plt
import math
import pandas as pd

class NeuralNetwork():
	
	def __init__(self,size,fname):
		random.seed(1)
		
		self.trainingSetIP =  numpy.genfromtxt(fname, delimiter=',', skip_header=0)
		self.IP = self.trainingSetIP[:,4]
		
		#Normalization
		self.s_min=numpy.amin(self.IP)
		print "Minimum: ",self.s_min
		self.s_max=numpy.max(self.IP)
		print "Maximum: ",self.s_max

		self.diff = self.s_max-self.s_min
		self.sub = self.s_min / float(self.diff)

		# range r, size = n -r
		self.r = 3
		self.size = size 

		self.synaptic_weights = 2 * random.random((self.r ,1)) -1

		
		self.IP = array([[self.IP[i-self.r+1+j]/self.diff - self.sub for i in range(self.r)]
						 for j in range(self.size - self.r)])
		
		self.IP = self.IP[(self.r-1):]

		self.op_size = self.size - (2*self.r) + 1
		self.OP = array([self.trainingSetIP[:self.op_size,1] for i in range(1)]).T
		self.OP = self.OP/self.diff - self.sub
		self.Xaxis = [i for i in range(self.op_size)] 
	
	def func_sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def func_sigmoid_derivative(self,x):
		return x * (1-x)

	def think(self, IP):
		return self.func_sigmoid(dot(IP, self.synaptic_weights))


	def train_network(self, IP, OP, iterations):
		for i in xrange(iterations):
			output = self.think(IP)

			error = OP - output
			#self.scatter_mean_error(error,i)
			#plt.scatter(i, abs(error[0]), s=4)
		
			adjustment = dot(IP.T , (error * self.func_sigmoid_derivative(output)) )

			self.synaptic_weights += adjustment
		plt.show()
		
	def scatter_mean_error(self,error,xvalue):
		min_mean_error = 1
		'''
		mean_error = 0.0
		for j in range(self.r):
			mean_error += error[j]
		mean_error /= 5
		if min_mean_error > mean_error:
			min_mean_error = mean_error
			iteration = xvalue
			print "Min for: " ,min_mean_error ," at " ,iteration
		'''

		plt.scatter(xvalue,error[xvalue], s=4)
		

if __name__ == "__main__":
	nn = NeuralNetwork(100,"aapl.csv")
	print "Generating initial synaptic weights:"
	print nn.synaptic_weights

	print "\nTraining data-set: Input:"
	print nn.IP
	print "Output:"
	print nn.OP
		
	#plt.plot(nn.Xaxis, nn.IP[:,9], label="Training I/P")
	plt.plot(nn.Xaxis, nn.OP, label="Training O/P")
	plt.legend()
	
	plt.show()

	nn.train_network(nn.IP, nn.OP, 4000)

	print "Final synaptic weights:"
	print nn.synaptic_weights
	
	
	nn1 = NeuralNetwork(1100,"aapl(2).csv")

	actualOP = array([0.0 for i in range(nn1.op_size)])
	plt.plot(nn1.Xaxis, nn1.OP , label="Expected output")
	for i in range(nn1.op_size):
		#print nn.IP[i]
		actualOP[i]= nn.think(nn1.IP[i])
	plt.plot(nn1.Xaxis, actualOP, label="Predicted output")
	#Percentage diff
	#plt.plot(nn1.Xaxis, abs(actualOP-nn1.OP)/nn1.OP, label="Achieved Efficiency")
	plt.legend()	

	vertical_lines = [i for i in range(0,2500,100)]
	#for vl in vertical_lines:
			#plt.axvline(x=vl, color='k', linestyle='--')

	plt.show()