#!/usr/bin/env python
# coding: utf-8

# # Fruit Recognition Project
# This fruit recognition project for the course EE482, and we consider it __Case-c__ and it was created by: <br>
# Khalid Alsubhi - 1847371 <br>
# Abdulrahman Aljuhani - 1847693 <br>
# ***
# The project can recognise three fruits, whuch are: [Apple, Tomato, Banana] and the model can be saved after training and continue the training again. <br>
# Originally, this project was supposed to only take apple and tomato, but we decide it to increase it since we thought tomato and apple are really similar and we want it to see how it will recognize another fruit like banana since it have a diffrent shape and color. we are happy with the results of the project since it achevied around 95% accuracy. we used bith Jupyter and google collab for computing power and easier to run code boxes instead of the whole code. <br>
# This code have <font color=green> Run</font> in the green color which means the code won't work without this code box, but <font color=orange> Not Important to Run</font> with the yellow color won't affect the model/ code. <br>
# At the end of the code, the project can test images and predict what fruit they are with a confidence percentage.<br>
# The location of the saved model will be in a folder named `\model` . <br>
# Refrences of this code and datasets are below which helped us to apply the idea of this project.<br>

# # Imports
# most importantly import numpy, os and tensorflow
# 
# <font color=green> Run</font>

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from PIL import image # For Resizing new Images

# # Resize
# This function will resize all images in the directory, we used this function because we want to reduce the overall size of images and compress them.
# 
# Thus function is not necessory to run since the datasets are already resized to 100 x 100, and it will be resize in the next function
# 
# <font color=orange> Not Important to Run</font>

# In[8]:






# Resize all of the photos of the data set to image size, this will make sure all images are 100 x 100 even new images
# f = 'Apple_vs_Tomato\\Tomatoes' # change this to the desired directory
# for file in os.listdir(f):
#     f_img = f+"/"+file
#     img = image.open(f_img)
#     img = img.resize((100,100)) # desired size
#     img.save(f_img)
# <font color=green> Run</font>

# In[2]:


image_size = (100, 100)
batch_size = 32


# # Training
# The dataset, with a split of 80% training and 20% validation which is used for testing, we used 123 as a seed for shuffling
# 
# <font color=green> Run</font>

# In[10]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Apple_vs_Tomato", # directery of the dataset
    validation_split=0.2,
    subset="training",
    seed=123, # fixed number for shuffling
    image_size=image_size,
    batch_size=batch_size,
)


# # Validation
# same as above but for validation
# 
# <font color=green> Run</font>

# In[11]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Apple_vs_Tomato", # directery of the dataset
    validation_split=0.2,
    subset="validation",
    seed=123, # fixed number for shuffling
    image_size=image_size,
    batch_size=batch_size,
)


# # Class Names
# to make sure we've got the correct clases, we view the avalible clases
# 
# <font color=orange> Not Important to Run</font>

# In[12]:


class_names = train_ds.class_names
print(class_names)


# # Visulize
# This will help us to visulize images of the train dataset and view the clases where
# 
# 
# 0 = Apple, 
# 1 = Tomato, 
# 2 = Banana.
# 
# <font color=orange> Not Important to Run</font>

# In[13]:


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(15):
        ax = plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# Same as above, visulize images of the validation dataset
# 
# 
# 0 = Apple, 
# 1 = Tomato, 
# 2 = Banana.
# 
# <font color=orange> Not Important to Run</font>

# In[14]:


plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    for i in range(15):
        ax = plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# # Data Augmantion 
# We use data augmantion to avoide overfitting, since originally the data will always overfit.
# 
# We randomlly flip the images horizontally and rotate it and zoom in.
# 
# <font color=green> Run</font>

# In[15]:


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ]
)


# Visulize the dataset images after data augmantion by choosing random class and view the first picture
# 
# <font color=orange> Not Important to Run</font>

# In[16]:


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(15):
        augmented = data_augmentation(images)
        ax = plt.subplot(3, 5, i + 1)
        plt.imshow(augmented[0].numpy().astype("uint8"))
        plt.axis("off")


# # Model Creation
# We have three models, at first we used the first model but the results were not sastifying since it had low accuracy.
# 
# The second model also had about 85% accuracy and we used dropout to reduce overfitting.
# 
# Finally the third model is our used model for the final training, it reaches 95% accuracy and avoid overfitting.
# 
# <font color=orange> Not Important to Run</font>

# In[17]:


height = image_size[0]
width = image_size[1]
model = Sequential([
    layers.Conv2D(16,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(class_names)),
])


# <font color=orange> Not Important to Run</font>

# In[17]:


height = image_size[0]
width = image_size[1]
model = Sequential([
    layers.Conv2D(16,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names)),
])


# The best model with 8 layers
# 
# <font color=green> Run</font>

# In[17]:


height = image_size[0]
width = image_size[1]
model = Sequential([
    layers.Conv2D(16,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names)),
])


# # Model Training
# Model parameters and choosing the loss function
# 
# <font color=green> Run</font>

# In[18]:


model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'],
             )


# This will let us to save the model to continue training later on since we don't want to lose the training when we shutdown the program or google collab session ends
# 
# <font color=green> Run</font>

# In[19]:


#save directiry 
checkpoint_path = 'model/model_cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
#save the model
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# This will load the model before training again and it evaluate the model before training it, but you can't run it if you don't have the model already!!
# 
# <font color=orange> Not Important to Run</font>

# In[20]:


# Loads the weights before training
model.load_weights(checkpoint_path)
# evaluate
loss, acc = model.evaluate(val_ds,verbose=2)
print('Untrained model, accuracy: {:5.2f}%'.format(100 * acc))


# The number of epochs for training iterations, originally it's 50, but for testing the code is set to 1
# 
# <font color=green> Run</font>

# In[21]:


epochs = 1


# # Start Training the Model
# <font color=green> Run</font>

# In[22]:



history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs= epochs,
    callbacks=[cp_callback]
)


# This will build the model and display a summery and list the saved model
# 
# <font color=green> Run</font>

# In[23]:


model.build()
model.summary()
# print the files of the directory
os.listdir(checkpoint_dir)


# This is the same as the above one, will load the new trained model and evaluate it
# 
# <font color=orange> Not Important to Run</font>

# In[24]:


# Loads the weights of the new model
model.load_weights(checkpoint_path)
# evaluate the trained model
loss, acc = model.evaluate(val_ds,verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))


# # Visulaze Training Model Results
# This code will Diplay both the accuracy and loss of the training and validation, ofcourse it won't be visable with an epoch of value 1.
# 
# <font color=orange> Not Important to Run</font>

# In[25]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# # Prediction
# finally, this will take images and predict what they are with a confidince percentage some of the images are taken by us

# In[26]:


for i in os.listdir('Apple_vs_Tomato_testing'):
    img = tf.keras.utils.load_img('Apple_vs_Tomato_testing\\'+i, target_size=image_size)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("The image \t{}\t is most likely a \t {}  \t with a {:.2f}% confidence."
        .format(i,class_names[np.argmax(score)], 100 * np.max(score))
    )


# In[ ]:


# References:
#
# Tensorflow documentation: https://www.tensorflow.org/tutorials/images/classification
#
# dataset from: https://www.kaggle.com/aelchimminut/fruits262
#
# dataset from: https://www.kaggle.com/databeru/classify-15-fruits-with-tensorflow-acc-99-6/data
#
# dataset from: https://www.kaggle.com/moltean/fruits

