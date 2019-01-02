#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary modules
import matplotlib as plt
import cv2
import pandas as pd
import numpy as np
import glob

#import necessary modules from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#import tensorflow backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten, BatchNormalization, Convolution2D , MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import to_categorical


# In[2]:


#Create a list of input images and take labels as input
training_list = glob.glob('/home/sarvesh/ML_Github/hbwid/training_set/*.jpg')
df = pd.read_csv('/home/sarvesh/ML_Github/hbwid/train.csv')
training_sorted = sorted([x for x in training_list])


# In[9]:


#Preprocess images from dataset

#encode string Ids into numbers by creating a new column 'Id_numeric'
le = LabelEncoder()
df['Id_numeric'] = le.fit_transform(df['Id'])

#sort images in lexicographic order
df_sorted = df.sort_values('Image')

#drop columns like 'Image' and 'Id' that have string data types 
df_sorted.drop(['Image', 'Id'], axis = 1, inplace = True)

#Select only the first 7000 rows
df_sorted = df['Id_numeric'][:7000].values
print(df_sorted[:5])
print(df_sorted.shape)


# In[10]:


#function to resize image 

def resize_image(image, inter = cv2.INTER_AREA):
    
    #specify dimensions of the resized image
    dim = (256, 256)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    
    #reshape the image and flatten it out into a single array
    #esized = np.reshape(resized, (65536, 1, 3))
    
    return resized


# In[11]:


#initialize an empty list to store final training images
training_images_final = []

#initialize counter to check the number of images added
count = 0

for i in training_sorted:
    
    #Create a dataset of only 7000 images
    if count == 7000:
        break
        
    #read in a color image (channel = 0 i.e. bit depth = 0) 
    img = cv2.imread(i, 0)
    
    #obtain resized image from the function in the above cell
    img = resize_image(img)
    
    #append to final list of training images
    training_images_final.append(img)
    
    #increment counter
    count = count + 1
    
    if count % 1000 == 0:
        print("{} images have been preprocessed and added to final dataset".format(count))


# In[12]:


#Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(training_images_final, df_sorted, 
                                                    test_size = 0.35, random_state = 42)
X_train = np.array(X_train)
X_val = np.array(X_val)
print(type(y_train))
print(X_train.shape)

#assign None to training_images_final to save memory
training_images_final = None


# In[13]:


# flatten 256*256 images to a 65536 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
print("{} {} {}".format(X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_val = X_val.reshape(X_val.shape[0], num_pixels).astype('float32')


# In[14]:


# one hot encode outputs
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
num_classes = y_val.shape[1]
print(y_train.shape)
print(y_val.shape)


# In[15]:


#Specify the lengths of the training and testing sets as well as the batch size
train_length = len(X_train)
val_length = len(X_val)
batch_size = 128


# In[ ]:


#Create Neural Network

#Specify the type of model being used
model = Sequential()

#Add layers to the model
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
#model.add(Dense(128, input_shape = (256, 256, 3)))  #As each image has resolution 256x256
#model.add(Dense(256, activation = "relu"))
#model.add(Convolution2D(64, (3, 3), activation = 'relu'))
#model.add(Convolution2D(64, (3, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Convolution2D(128, (3, 3), activation = 'relu'))
#model.add(Convolution2D(128, (3, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dense(512, activation = 'relu'))
#model.add(Dense(128, activation = 'softmax'))  #Softmax function needs one node for every output class

#Compile the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])

#review the model
print(model.summary())


# In[ ]:


# Optional approach :
#Here we take only 1000 images (small dataset) for the purpose of training and validation
#Generators are created to create different images from the same image
#with some sort of augmentation in terms of rotation, shearing or translation etc
#using ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    class_mode = "categorical")

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    class_mode = "categorical")


# In[ ]:


#Optional approach : 

#Prepare data from training and validation sets
train_generator = train_datagen.flow(np.array(X_train), y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), y_val, batch_size=batch_size)


# In[ ]:


#Optional approach :

#fit data to the generator
history = model.fit_generator(
    train_generator, 
    steps_per_epoch = train_length // batch_size,
    epochs=30,
    validation_data = validation_generator,
    validation_steps=  val_length // batch_size
)


# In[ ]:


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=batch_size, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_val, y_val, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

