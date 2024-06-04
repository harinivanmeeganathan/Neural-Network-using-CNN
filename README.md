# Neural-Network-using-CNN
Neural network with the use of convolutional layers using Keras


a. Architectures of the NNs, with figures for tasks 1 and 2.  
 
Neural Networks  : 
An artificial neural network (ANN), often known as a neural network or neural net, is a type of machine learning model that mimics the way the human brain works. It is built up from several levels of neurons or units that are interconnected. Pattern identification and decision-making are two areas where neural networks have proven extremely helpful. 

Characteristics of neural networks: 
Layers: Neural networks consist of layers, such as an input layer, hidden layers, and an output layer. The number of hidden layers and neurons in each layer vary 
depending on the architecture. 
Weights and Biases: There is a bias attached to each neuron, and each link between neurons has a weight. During training, the network's optimal values for these 
parameters are determined. 
Activation Functions: Neurons in each layer applies activation functions to the weighted sum of their inputs. Activation functions are sigmoid, ReLU (Rectified Linear Unit), and softmax. 
Building a  Neural Networks for supervised learning for Multi-class classification Problem. 
The Neural Network for the Model for Task 1 is as follows, 
 
Task 1: 
Model 1 
 
Input Layer – Flatten Layer ( Input Shape is (60000,28,28,1) which is (num_samples, height, width, channels) ) 
Hidden Layer : 
Layer 1 :  Dense Layer using ReLu activation and 20 neurons. 
Layer 2 :  Dense Layer using ReLu activation and 8 neurons. 
Layer 3 : Dense Layer using Sigmoid activation and 1 neuron. 
 
Output Layer:  Dense Layer with number of classes (10) as a neurons with Softmax activation.  
 
Model 2 
 
Input Layer – Flatten Layer ( Input Shape is (60000,28,28,1) which is (num_samples, height, width, channels) ) 
Hidden Layer : 
Layer 1 :  Dense Layer using ReLu activation and 10 neurons. 
Layer 2 :  Dense Layer using selu activation and 4 neurons. 
Layer 3 : Dense Layer using Sigmoid activation and 2 neurons. 
Layer 4 : Dense Layer using softsign activation and 40 neurons. 
Layer 5 : Dense Layer using softplus activation and 60 neurons. 
 
Output Layer:  Dense Layer with number of classes (10) as a neurons with Softmax activation.  

Model 3 
 
Input Layer – Flatten Layer ( Input Shape is (60000,28,28,1) which is (num_samples, height, width, channels) ) 
Hidden Layer : 
Layer 1 :  Dense Layer using ReLu activation and 10 neurons. 
Layer 2 :  Dense Layer using selu activation and 4 neurons. 
Layer 3 : Dense Layer using Sigmoid activation and 2 neurons. 
Layer 4 : Dense Layer using softsign activation and 40 neurons. 
Layer 5 : Dense Layer using softplus activation and 60 neurons. 
Layer 6 : Dense Layer using softsign activation and 300 neurons. 
 
Output Layer:  Dense Layer with number of classes (10) as a neurons with Softmax activation. 
 
 
 Model 4
 
Input Layer – Flatten Layer ( Input Shape is (60000,28,28,1) which is (num_samples, height, width, channels) ) 
Hidden Layer : 
Layer 1 :  Dense Layer using ReLu activation and 10 neurons. 
Layer 2 :  Dense Layer using selu activation and 4 neurons. 
Layer 3 : Dense Layer using Sigmoid activation and 2 neurons. 
Layer 4 : Dense Layer using softsign activation and 40 neurons. 
Layer 5 : Dense Layer using softplus activation and 60 neurons. 
Layer 6 : Dense Layer using softsign activation and 300 neurons. 
Layer 6 : Dense Layer using ReLu activation and 300 neurons. 
 
Output Layer:  Dense Layer with number of classes (10) as a neurons with Softmax activation.  
 
Convolution Neural Networks (CNN) 
A specialized form of neural network, the Convolutional Neural Network (CNN) is specifically engineered to handle data that resembles a structured grid, including time series or image data. In the domain of computer vision, CNNs exhibit notable efficacy due to their ability to autonomously acquire hierarchical features from unprocessed pixel data. 


Characteristics of Convolution Neural Networks (CNN) : 
 
Convolutional layers : Convolutional layers are utilized by CNNs to scan small regions of input data with kernels, which are learnable filters. This enables them to identify local features and patterns. 

 
Pooling layers: Pooling layers down-sample the feature maps subsequent to convolution in order to enhance translation consistency alongside decrease computational complexity. 
Flatten Layer: The flattened 1D vector resulting from the convolution and pooling layers is then transmitted through conventional fully interconnected layers. 
 
Task 2: 
 
Model 1 

Input Layer – Conv2D Layer ( Input Shape is (28,28,1) which is  (height, width, channels)) with 28 * 28 Pixel grayscale images. 
Hidden Layer : 
Layer 1 :  Convolutional Layer using ReLu activation and 128 filters and kernel size of (3*3). 
Layer 2 :  Convolutional Layer using ReLu activation and 128 filters and kernel size of (3*3). 
Layer 3 : MaxPooling2D Layer for down-sampling. 
Layer 4 : Flatten Layer for converting 2D vectors to 1S vector. 
Layer 5 : Dense Layer using ReLu activation and 512 neurons. 
Layer 6 : Dropout Layer using dropout rate of 0.2. 
 
Output Layer:  Dense Layer with number of classes (10) as a neurons with Softmax activation.  
 

Model 2 

Input Layer – Conv2D Layer ( Input Shape is (28,28,1) which is  (height, width, channels)) with 28 * 28 Pixel grayscale images. 
Hidden Layer : 
Layer 1 :  Convolutional Layer using ReLu activation and 512 filters and kernel size of (3*3). 
Layer 2 :  Convolutional Layer using ReLu activation and 512 filters and kernel size of (3*3). 
Layer 3 : MaxPooling2D Layer for down-sampling. 
Layer 4 : Flatten Layer for converting 2D vectors to 1S vector. 
Layer 5 : Dense Layer using ReLu activation and 1024 neurons. 
Layer 6 : Dropout Layer using dropout rate of 0.3. 
Layer 7 : Dense Layer using softmax activation and 10 neurons. 
Layer 8 : Dense Layer using softsign activation and 50 neurons. 
 
Output Layer:  Dense Layer with number of classes (10) as a neurons with sigmoid activation.  
 
b. The description of the optimizer and learning rate you applied in the final model of task 2 and the optimizer or change of learning rate you used in task 3.  
 
Task 2, I have used Stochastic Gradient Descent (SGD) Optimizer for Model 1 and Model 2. 
 
Stochastic Gradient Descent (SGD)  
 
Using Optimizer Stochastic Gradient Descent (SGD) is a widely employed optimization method within the fields of machine learning and deep learning. In contrast to conventional approaches to gradient descent that employ the complete dataset for every iteration, Stochastic Gradient Descent (SGD) employs a random selection process to utilize only one sample from the dataset in each iteration. This methodology proves to be especially advantageous in the context of handling extensive datasets, since it effectively decreases the duration required for training. Despite the potential noise introduced by randomness, the primary objective remains the attainment of the minimum. Consequently, stochastic gradient descent (SGD) proves to be a helpful technique for optimizing gradients efficiently when dealing with large datasets. 
 
Task 3, I have used Adam Optimizer for the best Model. 
 
Adam Optimizer  
The Adam optimization algorithm integrates concepts from both the RMSProp and Momentum methods. The algorithm calculates adaptive learning rates for each 
parameter and operates in the following manner. 
• Initially, the algorithm calculates the exponentially weighted average of previous gradients, denoted as vdW. 
• Furthermore, the algorithm calculates the exponentially weighted average of the squares of previous gradients, denoted as sdW. 
• Thirdly, the calculated averages exhibit a tendency towards zero, and in order to mitigate this bias, a bias correction is implemented using the variables 
vcorrecteddW and scorrecteddW. 
• Finally, the parameters are adjusted based on the data obtained from the computed averages. 
W = W – Alpha(vcorrecteddW/ scorrecteddW + e) 
where, 
Alpha is the learning rate and e is the small value to avoid dividing by the zero. 
 
Learning Rate: 
 
Learning rate is a crucial hyperparameter in CNNs and other machine learning models. Gradient-based optimization techniques modify parameters based on the learning rate. Model training depends on finding the right learning rate, which typically requires experimental testing. Increasing the learning rate may expedite convergence but may surpass the ideal solution. While a slower learning rate assures consistency, it may lengthen training. Using adaptive approaches or learning rate schedules, the learning rate can be optimized during training, balancing fast convergence with steady optimization. A hyper-parameter called learning rate(α) controls the speed at which an algorithm gathers or changes parameter estimate values. 
 
I have used the learning rate in my task 2 models 1 and 2 with a rate of 0.002, which scores an accuracy of 86.29% for the training dataset. Similarly, I have used the learning rate for task 3 at the best model rate of 0.003, which scores an accuracy of 94.59% for the training set. 
 
c. Experiments and performances, with parameter setting.  
 
Task 1 Model 1: 
• Architecture: A neural network with 3 hidden layers. 
• Parameters: 20, 8, and 1 neurons in the hidden layers, and a single output layer with Softmax activation. 
• Accuracy (Training): 20.08% 
• Accuracy (Testing): 20.19% 
• Loss Value: 1.9170 
Task 1 Model 2: 
• Architecture: A neural network with multiple hidden layers. 
• Parameters: Changing the numbers of neurons in the hidden layers, and a single output layer with Softmax activation. 
• Accuracy (Training): 32.00% 
• Accuracy (Testing): 31.87% 
• Loss Value: 1.5026 
Task 1 Model 3: 
• Architecture: A neural network with multiple hidden layers. 
• Parameters: Changing the numbers of neurons in the hidden layers, and a single output layer with Softmax activation. 
• Accuracy (Training): 55.04% 
• Accuracy (Testing): 54.99% 
• Loss Value: 1.3215 
Task 1 Model 4: 
• Architecture: A neural network with multiple hidden layers. 
• Parameters: Changing the number of neurons in the hidden layers, and a single output layer with Softmax activation. 
• Accuracy (Training): 65.32% 
• Accuracy (Testing): 64.53% 
• Loss Value: 1.0000 
Task 2 Model 1: 
• Architecture: A convolutional neural network (CNN) with convolutional and dense layers. 
• Parameters: Modifying the numbers of filters and neurons in the hidden layers. 
• Accuracy (Training): 86.29% 
• Accuracy (Testing): 85.08% 
• Loss Value: 0.4172 
Task 2 Model 2: 
• Architecture: A convolutional neural network (CNN) with convolutional and dense layers. 
• Parameters: Modifying the numbers of filters and neurons in the hidden layers. 
• Accuracy (Training): 36.86% 
 
• Accuracy (Testing): 36.69% 
• Loss Value: 2.2070 
Task 3 Best Model: 
Architecture: A convolutional neural network (CNN) with convolutional and dense layers. 
Parameters: Modifying the numbers of filters and neurons in the hidden layers. 
Accuracy (Training): 94.59% 
Accuracy (Testing): 91.35% 
Loss Value: 0.1920 
From the above findings, the nest model is task 3 model using the Adam optimizer 
with CNN architecture. 

d. Discussion on the improvement/deterioration of the NN’s performance after changing the architecture and parameter setting for each task and findings of comparing the results from all three tasks. 
 

 
From the above table observations,  
I have tried Four models in the task 1, where I found increasing the number of dense layers and neurons helped to increase the accuracy and reduces the loss value. Whereas, using the optimizer in the convolution neural networks helps to increase the accuracy by rapidly changing from a difference of 65% to 86%. Comparing the accuracy of all the models in the task 1 varied by increasing the dense layers and neuron in each model, which leads the rapid scoring of the accuracy in the Task 1 models. In task 2 of the model 2, there was a deterioration of accuracy of 36% with an increase in loss value of 2.2070 which is higher. It occurred due to the changes in the dropout layers and filters in convolution layers as well as variation in neurons in the dense layers.  For 
all the models, I have used the loss attribute of categorical_crossentropy to determine the loss value. 

The Best Model of my observation was utilizing the “ADAM” Optimizer which rapidly increased the accuracy of both training and testing datasets more than 90% (training of 94% and testing of 91%) and decreasing the loss value of  0.1920, the major key notice was the learning rate is increased with rate of 0.003 and the increase in the filters in convolution layers, and number of dense layers was increased which are mentioned earlier in the architecture figure. 
