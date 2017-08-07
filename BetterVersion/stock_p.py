from numpy import exp, array, random, dot
import numpy
import matplotlib.pyplot as plt
import math
import pandas as pd

class NeuralNetwork():
	
	def __init__(self,size,fname):
		random.seed(1)
		
		self.trainIP =  numpy.genfromtxt(fname, delimiter=',', skip_header=0)
		self.IP = self.trainIP[:,1]
		
		#Normalization
		self.s_min=numpy.amin(self.IP)
		print "Minimum: ",self.s_min
		self.s_max=numpy.max(self.IP)
		print "Maximum: ",self.s_max

		self.diff = self.s_max-self.s_min
		self.sub = self.s_min / float(self.diff)

		# range r, size = n -r
		self.r = 15
		self.size = size 

		self.synaptic_weights = 2 * random.random((self.r ,1)) -1

		
		self.IP = array([[self.IP[i-self.r+1+j]/self.diff - self.sub for i in range(self.r)]
						 for j in range(self.size - self.r)])
		
		self.IP = self.IP[(self.r-1):]


		self.op_size = self.size - (2*self.r) + 1
		self.OP = array([self.trainIP[:self.op_size,1] for i in range(1)]).T
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
			
			#plot error graph
			#plt.scatter(i, abs(error[0]), s=4)
		
			adjustment = dot(IP.T , (error * self.func_sigmoid_derivative(output)) )

			self.synaptic_weights += adjustment
		plt.show()