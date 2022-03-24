import csv
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf


class kohonen_network():
    def __init__(self, n, m, dim, iterations):
        self.n = n
        self.m = m
        self.alpha = 0.1
        self.dim = dim 
        self.iterations = iterations
        self.weights_vec = tf.Variable(tf.random.uniform([self.m * self.n, self.dim],minval=0.0, maxval=1.0))
        self.weight_label = np.array([])
        
    @tf.function
    def training_operator(self, vect_input):
        bmu_index = tf.argmax( tf.reduce_sum(tf.multiply(self.weights_vec, tf.stack(
            [vect_input for _ in range(self.m * self.n)])),1) )
        #print(bmu_index)
        paddings = [[bmu_index ,self.n*self.m -1 - bmu_index],[0, 0]]
        
        vect_input = tf.reshape(vect_input, shape = [1,self.dim])
        
        vect_input = tf.math.scalar_mul(self.alpha,vect_input)
        
        update = tf.pad(vect_input,paddings, mode = "CONSTANT",constant_values = 0.0)

        new_weightages_op = tf.add(self.weights_vec,update)
        for col in range(new_weightages_op.shape[1]):
            norm_col = tf.norm(tf.slice(new_weightages_op, [0, col], [new_weightages_op.shape[0], 1]))
            
            norm_vec = tf.fill([self.m*self.n,1],norm_col)
            slice_input = tf.slice(new_weightages_op,[0,col],[self.m*self.n,1])
            normalized_column = tf.divide(slice_input,norm_vec)         
            
            new_weightages_op = tf.transpose(new_weightages_op)
            normalized_column = tf.transpose(normalized_column)
            
            new_weightages_op = tf.tensor_scatter_nd_update(new_weightages_op,[[col]],normalized_column)
            
            new_weightages_op = tf.transpose(new_weightages_op)
            
        for row in range(new_weightages_op.shape[0]):
            norm_row = tf.norm(tf.slice(new_weightages_op, [row, 0 ], [1, new_weightages_op.shape[1]]))
            
            norm_vec = tf.fill([1,self.dim],norm_row)
            slice_input = tf.slice(new_weightages_op,[row,0],[1, new_weightages_op.shape[1]])
            normalized_row = tf.divide(slice_input,norm_vec)
            
            new_weightages_op = tf.tensor_scatter_nd_update(new_weightages_op,[[row]],normalized_row)
            
            
        self.weights_vec.assign(new_weightages_op)
        
    def train(self, input_vects,input_label):
        for col in range(input_vects.shape[1]):
             norm_col = tf.norm(tf.slice(input_vects, [0, col], [input_vects.shape[0], 1]))
             for i in range(input_vects.shape[0]):
                input_vects[i][col] /= norm_col
        for row in range(input_vects.shape[0]):
            norm_row = tf.norm(tf.slice(input_vects, [row, 0], [1, input_vects.shape[1]]))
            for i in range(input_vects.shape[1]):
                input_vects[row][i] /= norm_row
                 
        print("Start training...")
        # Training iterations
        for iter_no in range(self.iterations):
            print("Currently at: "+str(iter_no))
            # Train with each vector one by one
            for input_vect in input_vects:
                self.training_operator(input_vect)
        self.weightages = list(self.weights_vec.numpy())
        for weight in self.weightages:
            self.weight_label = np.append(self.weight_label,input_label[np.argmax (input_vects @ weight)])
        
        self.trained = True
        
        
    def test(self, output_vects,output_label):
        for col in range(output_vects.shape[1]):
             norm_col = tf.norm(tf.slice(output_vects, [0, col], [output_vects.shape[0], 1]))
             for i in range(output_vects.shape[0]):
                output_vects[i][col] /= norm_col
        count = 0
        for row in range(output_vects.shape[0]):
            if (  self.weight_label[np.argmax(self.weightages @ output_vects[row])] ==  output_label[row]):
                count+=1
        print("Arrcuracy: " + str(count/output_vects.shape[0]) )
if __name__ == '__main__':
    with open('wine.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        x = list(reader)
        x = x[1 : ]
        y = []
        for item in x:
            temp = item.pop(0)
            y.append(temp)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
        x_train = np.asarray(x_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.int)
        x_test = np.asarray(x_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.int)
        kn = kohonen_network(3,1,x_train.shape[1],100)
        
        #Training weight
        kn.train(x_train,y_train)
        
        kn.test(x_test,y_test)
        