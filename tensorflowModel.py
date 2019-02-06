import tensorflow as tf
from data_thing import train_test_data_creator
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train,x_test,y_train,y_test = train_test_data_creator(0.3)

n_nodes_hl1 = 500
n_nodes_hl2 = 100
n_nodes_hl3 = 1500

n_classes = 2
hm_epochs = 42

x = tf.placeholder('float')
y = tf.placeholder('float')
                
                
def deep_neural_network_model(data):
    
    hidden_1_layer = {'weight':tf.Variable(tf.random_normal([len(x_train[0]), n_nodes_hl1])),
                      'bias':tf.Variable(tf.zeros([n_nodes_hl1]))}

    hidden_2_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias':tf.Variable(tf.zeros([n_nodes_hl2]))}

    hidden_3_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'bias':tf.Variable(tf.zeros([n_nodes_hl3]))}

    output_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'bias':tf.Variable(tf.zeros([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_deep_neural_network(feature, label):
	prediction = deep_neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=label))
	myOptimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(hm_epochs):
			_,epochLoss = sess.run([myOptimizer,cost], {x:feature,y:label})
			print("epoch:", epoch, "loss: ", epochLoss)
	
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))

train_deep_neural_network(x_train,y_train)
	
	
	
	
	
