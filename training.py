import cv2 as cv
import numpy as np
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt


# 1) get data-set  already inside keras  _> call a function to load data 

(training_images,training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data()

# 2) normalize the data  pass values between 0 and 1 
training_images, testing_images = training_images/255, testing_images/255

# 3) show some images from the data-set
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


for i in range (16):
  plt.subplot(4,4,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(training_images[i],cmap=plt.cm.binary)
  plt.xlabel(class_names[training_labels[i][0]])
  
plt.show()

# 4) create the model
training_images=training_images[:10000]
training_labels=training_labels[:10000]
testing_images=testing_images[:2000]
testing_labels=testing_labels[:2000]

# 5) add layers to the model
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
#flattening means converting 2d array to 1d array
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
# 6) compile and train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# 7) fit the model
model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))
# 8) evaluate of the model
loss, acc = model.evaluate(testing_images, testing_labels)
print(f"loss: {loss}")
print(f"accuracy: {acc}")
# 9) save the model
model.save("cifar10_classifier.keras")

