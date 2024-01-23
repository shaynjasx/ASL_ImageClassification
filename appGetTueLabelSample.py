import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import random
class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

train = pd.read_csv('dataset/sign_mnist_train.csv')
test = pd.read_csv('dataset/sign_mnist_test.csv')

train_set = np.array(train, dtype = 'float32')
test_set = np.array(test, dtype = 'float32')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import asarray
from skimage.io import imread, imshow
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

modelCNN = tf.keras.models.load_model("CNN/cnn_model.h5")

#img = Image.open("image_input/B1.jpg")
img = Image.open("SAPredicted_Images/C.jpg")
image_input = asarray(img)
image_input = image_input/255.
image_input = resize(image_input, (28, 28, 1))
image_input = np.expand_dims(image_input, axis = 0)

prediction = modelCNN.predict(image_input)
st.write("prediction = ", prediction)

modelSA = tf.keras.models.load_model("SA/sa_model.h5")
predictionSA = modelSA.predict(image_input)
#               0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
MaxPositionLabel=np.argmax(prediction)  
MaxPositionindex=np.argmax(prediction, axis=1)  

st.write( "Index = ", MaxPositionLabel, " Predicted Label = ", class_names[MaxPositionindex[0]], " True Label = D")

indices = np.where(prediction>= 0.0001)
arr = prediction[prediction>= 0.0001]


indicesSA = np.where(predictionSA>= 0.0001)
arrSA = predictionSA[predictionSA>= 0.0001]


#arr = arr.sort(reverse=True)
for i in range(len(arr)):
    st.write("index : ", indices[1][i] , "value = ", class_names[indices[1][i]], " %0.2f%%" % (arr[i] * 100))

############################################################
#selection sort descending order using CNN
temp = 0
tempindex = 0
for i in range(0, len(arr)):
    for j in range(i+1, len(arr)):
        if arr[i] < arr[j]:
            temp = arr[i]
            tempindex = indices[1][i]
            
            arr[i] = arr[j]
            indices[1][i] = indices[1][j]
            
            arr[j] = temp
            indices[1][j] = tempindex
############################################################

############################################################
#selection sort descending order using SA
tempSA = 0
tempindexSA = 0
for i in range(0, len(arrSA)):
    for j in range(i+1, len(arrSA)):
        if arrSA[i] < arrSA[j]:
            tempSA = arrSA[i]
            tempindexSA = indicesSA[1][i]
            
            arrSA[i] = arrSA[j]
            indicesSA[1][i] = indicesSA[1][j]
            
            arrSA[j] = tempSA
            indicesSA[1][j] = tempindexSA
############################################################

###############################################
#create a graph using the predicted  accuracy data
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (12, 8))
plt.subplot(2, 2, 1)
plt.ylabel('Accuracy')
plt.xlabel('Letters')
data = []

if len(arr)<5:
    for i in range(len(arr)):
        langs = [class_names[indices[1][i]]]
        students = arr[i]
        percentage = " %0.2f%%" % (arr[i] * 100)
        plt.bar(langs,students, bottom=None, align='center', data=None)
        plt.text(langs,students,percentage, ha = 'center')
else:
        for i in range(5):
            langs = [class_names[indices[1][i]]]
            students = arr[i]
            percentage = " %0.2f%%" % (arr[i] * 100)
            plt.bar(langs,students, bottom=None, align='center', data=None)
            plt.text(langs,students,percentage, ha = 'center')
plt.legend()
plt.title('CNN Prediction Graph')

plt.subplot(2, 2, 2)
if len(arrSA)<5:
    for i in range(len(arrSA)):
        langs = [class_names[indicesSA[1][i]]]
        students = arrSA[i]
        percentage = " %0.2f%%" % (arrSA[i] * 100)
        plt.bar(langs,students, bottom=None, align='center', data=None)
        plt.text(langs,students,percentage, ha = 'center')
else:
        for i in range(5):
            langs = [class_names[indicesSA[1][i]]]
            students = arrSA[i]
            percentage = " %0.2f%%" % (arrSA[i] * 100)
            plt.bar(langs,students, bottom=None, align='center', data=None)
            plt.text(langs,students,percentage, ha = 'center')
plt.legend()
plt.title('CNN_SA Prediction Graph')
st.pyplot(fig)
############################################################################################

