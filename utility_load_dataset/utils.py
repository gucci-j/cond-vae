# coding: utf-8

import numpy as np
import sys
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def load_dataset(root_dir, folders, test_size=0.10, img_size=(28, 28, 3)):
	IMG_COLS, IMG_ROWS, IMG_CH = img_size[0], img_size[1], img_size[2]

	print('Acquiring images...')
	images = []
	labels = []
	if IMG_CH == 3:
		for folder in folders:        
			for filename in os.listdir(os.path.join(root_dir,folder)):
				if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.pgm', '.png', '.tiff', 'bmp']]):
					img = cv2.imread(os.path.join(root_dir, folder, filename), cv2.IMREAD_COLOR)
					if img is not None:
						image = cv2.resize(img, (IMG_COLS, IMG_ROWS))
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
						images.append(image)

						label = os.path.split(folder)[1].split("_")[1]
						labels.append(label)
	elif IMG_CH == 1:
		for folder in folders:        
			for filename in os.listdir(os.path.join(root_dir,folder)):
				if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.pgm', '.png', '.tiff', 'bmp']]):
					img = cv2.imread(os.path.join(root_dir, folder, filename), cv2.IMREAD_GRAYSCALE)
					if img is not None:
						image = cv2.resize(img, (IMG_COLS, IMG_ROWS))
						images.append(image)

						label = os.path.split(folder)[1].split("_")[1]
						labels.append(label)
	else:
		sys.exit('Unexpected input value!')
	
	print("Done!")
	print('>>> No. of images = %s. ' %  len(images))
	print('>>> No. of labels = %s. ' %  len(labels))
	print("Splitting dataset...")
	(x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=test_size)
	print("Done!")
	print("Converting to numpy array...")
	x_train, x_test = np.asarray(x_train), np.asarray(x_test)
	print("Done!")

	return (x_train, y_train), (x_test, y_test)
    
if __name__ == '__main__':
	img_size = (28, 28, 3)
	folders = [
		'category_0',
		'category_1',
		'category_2',
		'category_3',
		'category_4',
		'category_5',
		'category_6',
		'category_7',
		'category_8',
		'category_9'
	]
	(x_train, y_train), (x_test, y_test) = load_dataset(root_dir=r'/Users/gucci/Downloads/jaffe',
                                  folders=folders, test_size=0.10, img_size=img_size)
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0
	x_train = x_train.reshape((len(x_train),) + img_size)
	x_test = x_test.reshape((len(x_test),) + img_size)
    # 1-hot encoding
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)
	print('>>> Train Shape:', x_train.shape)
	print('>>> Test Shape:', x_test.shape)
