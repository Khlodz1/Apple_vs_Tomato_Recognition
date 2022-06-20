# Apple_vs_Tomato_Recognition
   The main idea of this project is that it can recognize three fruits which are apple, tomato and banana. The model of this project can be saved after training and continue training anytime again. However, the original idea of this project was to recognize apple and tomato, but we decide to add banana to them since we thought tomato and apple are really similar to each other. Also, we want to see how it will recognize another fruit like banana which has different shape and color. Nevertheless, we are totally happy with the results of this project since it achieved around 95% accuracy. 

   We used Jupyter and google collab to benefit form the power computing and it is also easier to run code in boxes instead of the whole code. We used the following libraries: Tensorflow, os and numpy. Tensorflow is our main library which is one of the most powerful libraries in machine learning that helped us to create the model which we decided to go with CNN. Our model consists of 8 layers which are as follow: 

CNN(16) – MaxPooling – CNN(32) – MaxPooling – CNN(64) – MaxPooling– CNN(128) – MaxPooling

   In addition, we apply data augmentation to avoid overfitting since we faced this problem in the beginning. We also print the predictions of images that is different from the dataset. Some of these images were taken by us. However, the code has more detailed comments for each block. 

   The datasets was taken from different sources to make our data bigger and the code was inspired by the following references:

Tensorflow documentation: https://www.tensorflow.org/tutorials/images/classification

dataset from: https://www.kaggle.com/aelchimminut/fruits262

dataset from: https://www.kaggle.com/databeru/classify-15-fruits-with-tensorflow-acc-99-6/data

dataset from: https://www.kaggle.com/moltean/fruit

