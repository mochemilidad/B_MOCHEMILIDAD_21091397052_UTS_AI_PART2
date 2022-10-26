#mochemilidad
#21091397052
#Multiple perceptron / Neuron batch and multiple layer 2

#inisialisasi numpy
import numpy as np

# inisialisasi variabel
# memasukan nilai variabel layer feature 10 dengan batch sejumlah 6
inputs = [
    [1.4, 3.7, 8.0, 2.7, 0.0, 3.5, 5.9, 3.3, 5.0, 5.0],
    [2.3, 3.1, 6.2, 7.6, 3.8, 3.1, 1.2, 2.4, 9.2, 7.4],
    [1.4, 9.5, 8.0, 2.5, 3.1, 0.1, 3.7, 7.1, 6.0, 0.5],
    [6.0, 3.4, 2.6, 7.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
    [1.4, 0.3, 7.2, 5.0, 8.2, 6.1, 9.2, 9.4, 27.3, 0.4],
    [3.2, 17.3, 4.5, 0.5, 3.1, 2.6, 1.7, 3.3, 9.1, 0.4]]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# memasukan jumlah weight sesuai dengan jumlah neuron yaitu sejumlah 5
weights1 = [
    [1.0, 4.8, 1.4, 2.5, 0.1, 3.5, 9.7, 4.5, 0.2, 5.5],
    [7.4, 9.7, 4.1, 2.4, 3.2, 8.4, 4.2, 4.4, 5.2, 5.4],
    [3.3, 6.1, 2.3, 1.9, 1.6, 3.2, 4.6, 4.8, 6.6, 5.8],
    [5.8, 4.3, 4.2, 7.8, 0.2, 7.4, 3.5, 0.7, 40.3, 1.1],
    [5.1, 1.7, 3.6, 2.7, 9.1, 1.3, 9.0, 0.7, 8.1, 3.1]]

# inisialisasi biases pada layer1 sesuai dengan neuron yang ditentukan yaitu layer 1 = 5 neuron
biases1 =   [4.7, 2.8, 1.0, 9.6, 3.1]

# inisialisasi jumlah weight 2, weight layer 2 = neuron layer 1 yaitu 5
# memasukkan jumlah weight sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [
    [0.3, 4.4, 2.9, 3.2, 1.2],
	[5.0, 1.3, 4.2, 7.5, 9.9],
	[0.1, 6.6, 3.0, 0.0, 3.7]]

# inisialisasi biases pada layer2 dengan neuron yang ditentukan yaitu 3
biases2 =  [8.2, 4.2, 5.6]


# output
# menghitung layer1 dengan (inputs*weight1) dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# menghitung layer2 dengan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs)