# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve , roc_auc_score
from sklearn.metrics import auc

from keras.models import model_from_json
from keras.models import load_model
from keras.utils import plot_model

def visualiseWeights():
 	
	model="model.json"
	json_file = open(model, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	orgmodel = model_from_json(loaded_model_json)
	# load weights into new model
	orgmodel.load_weights("model.h5")
	print("Loaded model from disk")
	orgmodel.summary()    
	l1=orgmodel.layers[6]   
	x1w=l1.get_weights()[0][:,:,0,:]
	for i in range(0,79):
		plt.subplot(9,9,i+1)
		plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")

	plt.show()	
	plot_model(model, to_file='model.png' , show_shapes=True)

	
	
	
	
visualiseWeights()


