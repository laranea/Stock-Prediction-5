from stock_p import NeuralNetwork
import matplotlib.pyplot as plt
import numpy
from numpy import array

if __name__ == "__main__":
	
	#input output to the network
	#NN(size of input, file)
	nn = NeuralNetwork(125,"stocks_data.csv")
	print "Generating initial synaptic weights:"
	print nn.synaptic_weights

	print "\nTraining data-set: Input:"
	print nn.IP
	print "Output:"
	print nn.OP
		
	plt.plot(nn.Xaxis, nn.OP, label="Training O/P")
	plt.legend()
	plt.show()

	
	#train the network
	nn.train_network(nn.IP, nn.OP, 4000)

	
	#training done
	print "Final synaptic weights:"
	print nn.synaptic_weights
	
	
	#test the network
	#nn1 new object
	nn1 = NeuralNetwork(2400,"stocks_data2.csv")


	plt.plot(nn1.Xaxis, nn1.OP , label="Expected output")
	
	actualOP = array([0.0 for i in range(nn1.op_size)])
	for i in range(nn1.op_size):
		actualOP[i]= nn.think(nn1.IP[i]) #think with s weights of prev object nn
	plt.plot(nn1.Xaxis, actualOP, label="Predicted output")
	plt.legend()	
	plt.show()

	#Percentage diff
	#plt.plot(nn1.Xaxis, abs(actualOP-nn1.OP)/nn1.OP, label="Achieved Efficiency")
	
	#vertical_lines = [i for i in range(0,2500,100)]
	#for vl in vertical_lines:
			#plt.axvline(x=vl, color='k', linestyle='--')

	