import pandas as pd 
import numpy as np 
import os 
import pdb 

#data_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/segment_features/'

#fold_dir = '../../LIFE_Codes/kfold_data/'


def combine_splits(data_dir):

	data_files = [x for x in os.listdir(data_dir) if x.endswith('.npy')]

	all_data = []
	full_data = dict()
	


	for i,dat_file in enumerate(data_files):
		data = np.load (os.path.join(data_dir, dat_file), allow_pickle=True).item()
	
		if i == 0: # Populate the dictonary
			for keys, values in data.items():
				if not isinstance (values, list):
					full_data[keys] = values.tolist()

		else:
			for keys,values in data.items():
				full_data[keys].extend (values)

		#print (len(data['turn_filename']))
	
	return full_data


def get_split_indices (fold_dir, full_data, fold_no=0):

	folds= np.array(os.listdir(fold_dir)) 
	fold_file = folds[folds  == 'split_' + str(fold_no)+'.npy'][0]
	

	data= np.load(os.path.join(fold_dir, fold_file ), allow_pickle=True).item()
	x_train,x_valid,x_test=data['x_train'],data['x_valid'],data['x_test']
	
	#keys = np.array(full_data['turn_filename']) 
	keys= np.array(full_data['family'])
	tr_indices = np.array([np.where(keys==x)[0] for x in x_train if x in keys]).flatten()

	vl_indices = np.array([np.where(keys==x)[0] for x in x_valid if x in keys]).flatten()

	ts_indices = np.array([np.where(keys==x)[0] for x in x_test if x in keys]).flatten()

	return tr_indices, vl_indices, ts_indices

'''
def main ():

	full_data = combine_splits(data_dir)

	
	#----Put this line in a  loop for k-fold experiment ------#
	tr_indices, vl_indices, ts_indices= get_split_indices(fold_dir, full_data, fold_no=2)


if __name__== "__main__":
	main()

'''
