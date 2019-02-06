import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def train_test_data_creator(size_rate=0.1):
	data = pd.read_csv("breast-cancer-wisconsin.data.txt")
	data.drop("id",1,inplace=True)
	data.replace("?",-99999,inplace=True)
	shuffle(data)
	y_received =[]
	for i in data["class"]:
		if(i==2):
			y_received.append([1,0])
		else:
			y_received.append([0,1])
	y_received  = np.array(y_received )
	return np.array(data[:int(len(data) * size_rate)]), np.array(data[:-int(len(data) * size_rate)]), y_received[:int(len(data) * size_rate)], y_received[:-int(len(data) * size_rate)]



if (__name__=="__main__"):
	print("program is executing")
