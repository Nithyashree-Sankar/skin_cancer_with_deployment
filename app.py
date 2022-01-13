from flask import Flask, render_template, request
import pandas
import pickle
import cv2
import numpy as np
import pandas as pd
import gzip
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.preprocessing import image




app = Flask(__name__)

#model = pickle.load(open('model.pkl', 'rb'))
#print("+"*50, "Model is loaded")
#with gzip.open('model.pkl', 'rb') as ifp:
#   print(pickle.load(ifp))
#model = keras.models.load_model(r'C:\Users\Nithyashree\Desktop\MINIPROJECT\model1.pb')
def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData
df=pd.read_csv("hmnist_28_28_RGB.csv")
fractions=np.array([0.8,0.2])
df=df.sample(frac=1)
train_set, test_set = np.array_split(
    df, (fractions[:-1].cumsum() * len(df)).astype(int))
x_test=test_set.drop(columns=['label'])


x_test=np.array(x_test).reshape(-1,28,28,3)
GRAPH_PB_PATH = 'saved_model_twoclasses.pb'
with tf.compat.v1.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       model11 = tf.compat.v1.GraphDef()

model=keras.models.load_model("model_vgg19.h5")
@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["GET","POST"])
def prediction():

	img = request.files['img']
	if img.filename != '':
		img.save(img.filename)
		img.read()
	#	empPicture = convertToBinaryData(img.filename)
		
	#with open(img,'rb') as f:
	#	contents=f.read()
	#contents = contents.rstrip("\n").decode("utf-16")
	#contents = contents.split("\r\n")
	#imgg = image.load_img(contents,target_size=(224,224))
	
	image1=Image.open(img)
	image1=image1.resize((224,224))
	
	x=np.asarray(image1)
	print(x.shape)
	x=x/255
	x=np.expand_dims(x,axis=0)
	img_data = preprocess_input(x)
	print(img_data.shape)
	a=np.argmax(model.predict(img_data), axis=1)
	s1=" "
	if(a==1):
		s1="Uninfected"
	else:
		s1="Infected"
	
	
	#imgs=x_test[1]
	
	#imgs=np.expand_dims(img,axis=0)
	#pred=model.predict(test)
	#print(pred[0])


	#img.save("img.jpg")
	#image = keras.preprocessing.image.load_img("img.jpg",target_size=(28,28))
	#bigger = cv2.resize(image, (28,28))
	#gray = cv2.cvtColor(bigger, cv2.COLOR_BGR2GRAY)
	#images = np.reshape(image, (1,28,28,3))
	#data = np.asarray(image.resize((28,28)))
	#x_orig=data.to_numpy()
	#x_orig = np.stack(data, axis=0)
	#x_orig=x_orig/255
	
	#x = keras.preprocessing.image.img_to_array(image)
	#x=x/255
	#x_orig = np.expand_dims(x_orig, axis=0)
	#images = np.vstack([x])
	#pred = model.predict([x_orig])
	#pred=np.arg(model.predict,axis=1)

	

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#image = cv2.resize(image, (28,28))

	#image = np.reshape(image, (1,224,224,3))

	#pred = model.predict(x)

	#pred = np.argmax(pred)

	#pred = labels[pred]

	return render_template("prediction.html", data=s1)


if __name__ == "__main__":
	app.run(debug=True)