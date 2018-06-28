# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve , roc_auc_score
from sklearn.metrics import auc 
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D , Conv2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import plot_model

from sklearn.ensemble import RandomForestClassifier

	
def filter( a , filter , array ):
	for  i in range(len(a[0])):
		a[filter][i] = array

	return a

	
def build_Gabor():
	filters = []
	ksize = 3
	for theta in np.arange(0, np.pi, np.pi / 7):
		for lamda in np.arange(0, np.pi, np.pi/4): 
			kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
			kern /= 1.5*kern.sum()
			filters.append(kern)
	return filters
	
	
def visualiseWeights():
    	# load full model	
    model="model.json"
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    orgmodel = model_from_json(loaded_model_json)

	# load weights into new model
    orgmodel.load_weights("model.h5")
    print("Loaded model from disk")
    orgmodel.summary()    
    l1=orgmodel.layers[0]   
    x1w=l1.get_weights()[0][:,:,0,:]
    for i in range(0,19):
        plt.subplot(5,4,i+1)
        plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
    plt.show()	
#%%
x = np.load("s2003_data.npy")
y = np.load("s2003_lables.npy")
print(x.shape)
print(y.shape)
x = np.append(x , np.load("s2004_data.npy") , axis=0)
print(x.shape)
x = np.append(x , np.load("full_data.npy") , axis=0)
print(x.shape)
y = np.append(y , np.load("s2004_lables.npy") , axis=0)
print(y.shape)
y = np.append(y , np.load("full_lables.npy") , axis=0)

print(y.shape)
# Y_list=[]
# for i in range(len(y)):
	# Y_list.append((y[0][0]))


# Y = np.array(Y_list)	
	
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

#%%
## Defining the model
input_shape=x[0].shape
					
model = Sequential()

model.add(Conv2D(20, (4),input_shape=input_shape ))
# model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(40, (3) ))
# model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(60, (3) ))
# model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(80, (2) ))
# model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(2 , activation = "softmax" ))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
			  
### set weights

print("setting weights: layer 1")
weights=model.layers[0].get_weights()
print(weights[0])
w = weights[0]
print(w.shape)
w = np.swapaxes(w , 0 , 3)
print(w.shape)
w = np.swapaxes(w , 1 , 2)
print(w.shape)
w = np.swapaxes(w , 2 , 3)
print(w.shape)
print(w)

conv1_0 = np.array([[0 ,-1/4, -1/4 , 0],
					[0 ,1/4, 1/4 , 0],
					[0 ,1/4, 1/4 , 0],
					[0 ,-1/4, -1/4 , 0]])
w = filter(w , 0 , conv1_0)


conv1_1 = np.array([[0 ,0, 0 , 0],
					[-1/4 ,1/4, 1/4 , -1/4],
					[-1/4 ,1/4, 1/4 , -1/4],
					[0 ,0, 0 , 0]])
w = filter(w , 1 , conv1_1)

conv1_2 = np.array([[-1/2 ,0, 0 , 0],
					[0 ,1/2, 0 , 0],
					[0 ,0, 1/2 , 0],
					[0 ,0, 0 , -1/2]])
w = filter(w , 2 , conv1_2)

conv1_3 = np.array([[0 ,0, 0 , -1/2],
					[0 ,0, 1/2 , 0],
					[0 ,1/2, 0 , 0],
					[-1/2 ,0, 0 , 0]])
w = filter(w , 3 , conv1_3)

conv1_4 = np.array([[-1/2 ,0, 0 , -1/2],
					[0 ,1/2, 1/2 , 0],
					[0 ,1/2, 1/2 , 0],
					[-1/2 ,0, 0 , -1/2]])
w = filter(w , 4 , conv1_4)

conv1_5 = np.array([[0 ,-1/2, -1/2 , 0],
					[-1/2 ,1/2, 1/2 , -1/2],
					[-1/2 ,1/2, 1/2 , -1/2],
					[0 ,-1/2, -1/2 , 0]])
w = filter(w , 5 , conv1_5)

conv1_6 = np.array([[-1/2 ,0, 0 , -1/2],
					[0 ,1/2, 1/2 , 0],
					[0 ,0, 0 , 0],
					[0 ,0, 0 , 0]])
w = filter(w , 6 , conv1_6)

conv1_7 = np.array([[0 ,0, 0 , 0],
					[0 ,0, 0 , 0],
					[0 ,1/2, 1/2 , 0],
					[-1/2 ,0, 0 , -1/2]])
w = filter(w , 7 , conv1_7)

conv1_8 = np.array([[-1/2 ,0, 0 , 0],
					[0 ,1/2, 0 , 0],
					[0 ,1/2, 0 , 0],
					[-1/2 ,0, 0 , 0]])
w = filter(w , 8 , conv1_8)

conv1_9 = np.array([[0 ,0, 0 , -1/2],
					[0 ,0, 1/2 , 0],
					[0 ,0, 1/2 , 0],
					[0 ,0, 0 , -1/2]])
w = filter(w , 9 , conv1_9)

conv1_10 = np.array([[0 ,-1/2, 0 , 0],
					[0 ,1/2, 0 , 0],
					[0 ,0, 1/2 , 0],
					[0 ,0, -1/2, 0]])
w = filter(w , 10 , conv1_10)

conv1_11 = np.array([[0 ,0, -1/2 , 0],
					[0 ,0, 1/2 , 0],
					[0 ,1/2, 0 , 0],
					[0 ,-1/2, 0 , 0]])
w = filter(w , 11 , conv1_11)

conv1_12 = np.array([[0 ,0, 0 , 0],
					[-1/2 ,1/2, 0 , 0],
					[0 ,0, 1/2 , -1/2],
					[0 ,0, 0 , 0]])
w = filter(w , 12 , conv1_12)

conv1_13 = np.array([[0 ,0, 0 , 0],
					[0 ,0, 1/2 , -1/2],
					[-1/2 ,1/2, 0 , 0],
					[0 ,0, 0 , 0]])
w = filter(w , 13 , conv1_13)

conv1_14 = np.array([[1/2 ,0, 0 , 1/2],
					[0 ,0, 0 , 0],
					[0 ,0, 0 , 0],
					[1/2 ,0, 0 , 1/2]])
w = filter(w , 14 , conv1_14)

conv1_15 = np.array([[0 ,1/2, 1/2 , 0],
					[1/2 ,0, 0 , 1/2],
					[1/2 ,0, 0 , 1/2],
					[0 ,1/2, 1/2 , 0]])
w = filter(w , 15 , conv1_15)

conv1_16 = np.array([[1/2 ,1/2, 1/2 , 1/2],
					[1/2 ,0, 0 , 1/2],
					[1/2 ,0, 0 , 1/2],
					[1/2 ,1/2, 1/2 , 1/2]])
w = filter(w , 16 , conv1_16)


conv1_17 = np.array([[0 ,0, 0 , 0],
					[0 ,1/2, 1/2 , 0],
					[0 ,1/2, 1/2 , 0],
					[0 ,0, 0 , 0]])
w = filter(w , 17 , conv1_17)

# conv1_18 = np.array([[1 ,1, 1 , 1],
					# [1 ,1, 1 , 1],
					# [1 ,1, 1 , 1],
					# [1 ,1, 1 , 1]])
# w = filter(w , 18 , conv1_18)
# conv1_19 = np.array([[1 ,1, 1 , 1],
					# [1 ,1, 1 , 1],
					# [1 ,1, 1 , 1],
					# [1 ,1, 1 , 1]])
# w = filter(w , 19 , conv1_19)

w = np.swapaxes(w , 0 , 3)
print(w.shape)
w = np.swapaxes(w , 1 , 2)
print(w.shape)
w = np.swapaxes(w , 0 , 1)
print(w.shape)
print(w)
print(weights[0])
print(w)
weights[0] = w 

model.layers[0].set_weights(weights)



# print("setting wights: layer 2")
# weights=model.layers[2].get_weights()
# w = np.array(weights[0])

# print(w.shape)
# w = np.swapaxes(w , 0 , 3)
# print(w.shape)
# w = np.swapaxes(w , 1 , 2)
# print(w.shape)
# w = np.swapaxes(w , 2 , 3)
# print(w.shape)

# filt = build_Gabor()

# conv2_0 = np.array([[1/16 ,2/16, 1/16 ],
					# [2/16 ,4/16, 2/16 ],
					# [1/16 ,2/16, 1/16 ]])				
# for i in range(len(filt)):
	# if i % 4 == 0:
		# w = filter(w , i , conv2_0)
	# else:
		# w[i] = np.array(filt[i])


# conv2_0 = np.array([[1/16 ,2/16, 1/16 ],
					# [2/16 ,4/16, 2/16 ],
					# [1/16 ,2/16, 1/16 ]])
# w = filter(w , 0 , conv2_0)

# conv2_4 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 4 , conv2_4)

# conv2_8 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 8 , conv2_8)

# conv2_12 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 12 , conv2_12)

# conv2_16 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 16 , conv2_16)

# conv2_20 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 20 , conv2_20)

# conv2_24 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 24 , conv2_24)

# conv2_28 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 28 , conv2_28)


# for i in range(29 , 39):
	# conv2_33 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
	# w = filter(w , i , conv2_33)


# print(w.shape)
# w = np.swapaxes(w , 0 , 3)
# print(w.shape)
# w = np.swapaxes(w , 1 , 2)
# print(w.shape)
# w = np.swapaxes(w , 0 , 1)
# print(w.shape)

# weights[0] = w 

# model.layers[2].set_weights(weights)


# print("setting wights: layer 3")
# weights=model.layers[4].get_weights()
# w = np.array(weights[0])
# print(w)
# print(w.shape)
# w = np.swapaxes(w , 0 , 3)
# print(w.shape)
# w = np.swapaxes(w , 1 , 2)
# print(w.shape)
# w = np.swapaxes(w , 2 , 3)
# print(w.shape)

# print(w[0])

# conv2_0 = np.array([[0 ,0, 0 ],
					# [0 ,0, 0 ],
					# [0 ,0, 0 ]])
# w = filter(w , 0 , conv2_0)

# print(w[0])

# print(w.shape)
# w = np.swapaxes(w , 0 , 3)
# print(w.shape)
# w = np.swapaxes(w , 1 , 2)
# print(w.shape)
# w = np.swapaxes(w , 0 , 1)
# print(w.shape)
# print(w)

# weights[0] = w 

# model.layers[4].set_weights(weights)



# print("setting wights: layer 4")
# weights=model.layers[6].get_weights()
# w = np.array(weights[0])
# print(w)
# print(w.shape)
# w = np.swapaxes(w , 0 , 3)
# print(w.shape)
# w = np.swapaxes(w , 1 , 2)
# print(w.shape)
# w = np.swapaxes(w , 2 , 3)
# print(w.shape)

# print(w[0])

# conv2_0 = np.array([[0 ,0 ],
					# [0 ,0 ]])
# w = filter(w , 0 , conv2_0)

# print(w[0])

# print(w.shape)
# w = np.swapaxes(w , 0 , 3)
# print(w.shape)
# w = np.swapaxes(w , 1 , 2)
# print(w.shape)
# w = np.swapaxes(w , 0 , 1)
# print(w.shape)
# print(w)

# weights[0] = w 

# model.layers[6].set_weights(weights)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Viewing model

model.summary()
plot_model(model, to_file='model.png' , show_shapes=True)

results = model.fit(
 x_train, y_train,
 epochs= 150,
 validation_data = (x_test, y_test)
)

#%%
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
np.save("no_acc.npy", np.array(results.history['acc']))
np.save("no_val_acc.npy", np.array(results.history['val_acc']))
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
np.save("no_loss.npy",	 np.array(results.history['loss']))
np.save("no_val_loss.npy", np.array(results.history['val_loss']))

# Evaluating the model

# Final evaluation of the model
scores=model.predict_proba(x_test)[:,1]
#print(scores)	
fpr, tpr, thresholds = roc_curve(y_test[:,1], scores, pos_label=1)
auc=auc(fpr, tpr)
f=plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC curve")
print("{}-->{}".format("ROCcurve", auc))
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
# plt.close()
print(eer)



score = model.evaluate(x_test, y_test,  verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = x[0:1]
print (test_image.shape)


		
# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))
print(y[0])

#%%

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=0
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]      
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.png')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.png')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(x_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
y_pred = model.predict_classes(x_test)
print(y_pred)
target_names = ['same', 'different']
					
# print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

# print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
# plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                     # title='Normalized confusion matrix')
plt.figure()
plt.show()

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# weights=loaded_model.layers[4].get_weights()
# w = np.array(weights[0])
# print(w)

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')

visualiseWeights()