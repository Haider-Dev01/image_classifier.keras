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

model= models.load_model('cifar10_classifier.keras')

#5) make a prediction on a single image
img=cv.imread('horse.jpg')
#6) COLOR CONVERSION BGR->RGB black and white vers colorful 
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

#7)greyscale conversion
plt.imshow(img, cmap=plt.cm.binary)

# 8) predict the image
prediction=model.predict(np.array([img])/255)
index=np.argmax(prediction) 
print(f"predicted class is : {class_names[index]}")
