import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import crop
from sklearn.utils import shuffle


PATH = os.getcwd()
# Define data path
data_path = r'D:\s1003828_HillerstromFHJ\Spring2004'
img_list = os.listdir(data_path)


data = np.loadtxt("data.txt" , dtype = str)



img_rows=128
img_cols=128
num_channel=2

img_data_list=[]
names = []
r_eye_x = []
l_eye_x = []
r_eye_y = []
l_eye_y = []
# for img in img_list:
	# input_img=cv2.imread(data_path + '/'+ img )
	# input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	# input_img_resize=cv2.resize(input_img,(128,128))
	# img_data_list.append(input_img_resize)
	# names.append(img)




# print(name)
for img in img_list:
	for i in range(0 , len(data)):
		if img == data[i][0]:
			if data[i][5]=='C' and data[i][6]=='N':
				input_img=cv2.imread(data_path + '/'+ img )
				input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
				names.append(img)
				input_img = crop.CropFace(input_img, (int(float(data[i][10])) , int(float(data[i][11])) ) , (int(float(data[i][8])) , int(float(data[i][9]))) , offset_pct=(10,10,10,40))
				input_img_resize=cv2.resize(input_img,(31,39))
				img_data_list.append(input_img_resize)
				


name = np.array(names)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

# for i in range(0 , len(img_data)):
	# img_data[i] = crop.CropFace(img_data[i], (l_eye_x[i] , l_eye_y[i] ) , (r_eye_x[i] , r_eye_y[i]) , offset_pct=(10,10,10,40))
	
print(name.shape)



img_data= np.expand_dims(img_data, axis=1) 
print (img_data.shape)

name1 , name2 = np.split(name , 2)
img_data1 , img_data2 = np.split(img_data , 2)
print(name1.shape)
print(img_data1.shape)
shuf_name2,shuf_data2 = shuffle(name2,img_data2, random_state=2)
print(shuf_name2.shape)
print(shuf_data2.shape)
shuf_data = np.vstack((img_data1 , shuf_data2))
shuf_names = np.hstack((name1 , shuf_name2))
print(shuf_names.shape)
print(shuf_data.shape)
chan1_data =[]
chan2_data = []

lables = np.zeros([ int(len(img_data)/2) , 2 ])


for i in range(0 , len(img_data)):
	if i%2 == 0:
		chan1_data.append(shuf_data[i])
		if shuf_names[i][:6] == shuf_names[i+1][:6]:
			lables[int(i/2)]=[1,0]
		else:
			lables[int(i/2)]=[0,1]
	else:	
		chan2_data.append(shuf_data[i])
		
		
chan1 = np.array(chan1_data)
chan2 = np.array(chan2_data)
print(chan1.shape)
print(chan2.shape)

#np.concatenate((img_data, img_data_2), axis=1)
img_data_fin = np.append(chan1, chan2 , axis=1)
#np.stack((img_data, img_data_2))

print (img_data.shape)

#%%
# Assigning Labels

# Define the number of classes


print(lables)

	  

# Shuffle the dataset
x,y = shuffle(img_data_fin,lables, random_state=2)
print(x.shape)
print(y)

np.save("s2004_data.npy", x)
np.save("s2004_lables.npy", y)

# Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
