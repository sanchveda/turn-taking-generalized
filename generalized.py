
#----------------This file is to generate labels for generalized continuous turn taking model ----------------#
import h5py
import numpy as np 
import os , csv 
import pdb 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd 
#import xlsxwriter
from matplotlib.ticker import PercentFormatter
plt.rcParams.update({'font.size': 14})

import pickle5 as pickle 

from joblib import Parallel, delayed
import multiprocessing

import seaborn as sns 

class_dict = {0: 'Aggressive' , 1:'Dysphoric' , 2:'Positive', 3:'other'}
root_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_continuous_combined_data/'

words_dir = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/Data_From_CMU/forced_alignment/Xlsx Files/'
wordfiles = [x for x in os.listdir(words_dir) if x.endswith('.xlsx')]


def load_text_data():
	data = np.load ('text_data_apo_full.npy',allow_pickle=True).item()
	
	family= data ['family'] 
	start_time=data ['start_time']
	end_time=data ['end_time'] 
	duration=data ['duration'] 
	subject=data ['subject'] 
	text=data ['text'] 
	sequence= data['sequence']

	return [family, start_time, end_time, duration, subject, text, sequence]




def get_proportion(t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration, c_duration):


	arr1= np.zeros_like(c_start) #------Number of overlaps + inside per construct 

	arr2= np.zeros_like(c_start) #--------Number of turns strictly in the window --

	arr3= np.zeros_like(c_start) #------Number of tutrns that wraps the construct
	

	

	neg1 = np.zeros_like(c_start)
	neg2 = np.zeros_like(c_start)
	for idx , (start_c, end_c, speaker_c) in enumerate(zip(c_start,c_end,c_speaker)):
		for jj, (start_t , end_t , speaker_t, duration ) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

			if speaker_c == speaker_t :
				# Case 1: All turns which fall in the construct interva even if it is partial +plus those that wrap
				if (start_t< start_c and end_t > start_c and end_c > end_t) : 
					arr1[idx]  +=  end_t - start_c 

				elif (start_t < end_c and end_t > end_c and start_c < start_t):
					arr1[idx] +=  end_c - start_t 

				elif (start_t >= start_c and end_t <= end_c) :
					arr1[idx] += end_t - start_t 

				elif (start_t < start_c and end_t > end_c) :
					arr1[idx] += end_c- start_c 
					
					
				
			
			

			elif speaker_c != speaker_t :
				if (start_t< start_c and end_t > start_c and end_c > end_t) : 
					arr2[idx]  +=  end_t - start_c 

				elif (start_t < end_c and end_t > end_c and start_c < start_t):
					arr2[idx] +=  end_c - start_t 

				elif (start_t >= start_c and end_t <= end_c) :
					arr2[idx] += end_t - start_t 

				elif (start_t < start_c and end_t > end_c) :
					arr2[idx] += end_c - start_c 
				
	

	arr3 = c_duration - ( arr1 +  arr2 )
	
	arr4 = arr1 / c_duration
	arr5 = arr2 / c_duration


	indices = ~np.isnan(arr4)  & ~np.isnan(arr5)

	arr1= arr1[indices]
	arr2= arr2[indices]
	arr3= arr3[indices]
	arr4= arr4[indices]
	arr5= arr5[indices]
	

	return  arr1,arr2,arr3 , arr4, arr5 , indices



def get_overlap (start_t, end_t, start_c, end_c, speaker_t, speaker_c, duration_t, duration_c, mode='case2'):
	duration = 0


	if mode == 'case2':
		if start_c < start_t and end_c > start_t and end_t> end_c:
			duration = end_c - start_t 
		elif start_c<end_t and end_c > end_t and start_t < start_c:
			duration = end_t - start_c
		elif start_c >= start_t and end_c <= end_t:
			duration = end_c - start_c 
		elif start_c < start_t and end_c> end_t:
			duration = end_t - start_t 

	return duration

def get_majority_construct (t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration,c_duration, label):

	arr2= np.zeros_like(t_start)
	label2= np.full(arr2.shape, -1)


	majority = np.zeros_like(t_start)
	for idx, (start_t , end_t , speaker_t, duration ) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

		for jj , (start_c, end_c, speaker_c, duration_c, construct) in enumerate(zip(c_start,c_end,c_speaker, c_duration, label)):

			if speaker_c== speaker_t:

				d = get_overlap (start_t, end_t , start_c , end_c , speaker_t, speaker_c, duration, duration_c)

				if d > arr2[idx]:
					arr2[idx] = d 
					label2[idx] = construct
					
	return arr2, label2

 	
def get_all_utterance (t_start, t_end, c_start, c_end, t_speaker, c_speaker, t_duration, c_duration, t_text):
	arr1= np.zeros_like(c_start)
	arr= []

	mark= np.full(arr1.shape, -1)
	
	for idx , (start_c, end_c, speaker_c) in enumerate(zip(c_start,c_end,c_speaker)):

		utterance = ""
		for jj, (start_t , end_t , speaker_t, duration , text) in enumerate(zip(t_start, t_end, t_speaker, t_duration, t_text)):
			if speaker_c == speaker_t :
				# Case 1: All turns which fall in the construct interva even if it is partial +plus those that wrap
				if (start_t< start_c and end_t > start_c and end_c > end_t)\
					or (start_t < end_c and end_t > end_c and start_c < start_t) \
					or (start_t >= start_c and end_t <= end_c)\
					or (start_t < start_c and end_t > end_c) :
						arr1[idx] = arr1[idx] +1 
						try:
							utterance += text + " "	
						except:
							pdb.set_trace()


		
		#arr= np.vstack [arr, utterance] if arr is not None else utterance
		if utterance != "":
			mark[idx]=len(utterance)
		arr.append(utterance.rstrip())
	arr= np.array(arr)
	
	return mark, arr  



def plot_histogram (x_label, y_label, title, name,   data):

	#bins = np.arange(0,np.max(arr)+1)
	#bins = np.arange(0,np.max(data)+1).astype(int)
	#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
	x= np.arange( 0, np.max(data)+1).astype(int)

	y= np.zeros_like(x).astype('float')
	for idx in x:
		y[idx]=np.sum (data == idx) / len(data)
	
	#bins = len(np.unique(data))
	#plt.hist(data, bins=bins)
	plt.bar (x=x, height=y, tick_label=list(x))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.ylim(0.0,0.6)
	#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
	plt.title(title)
	plt.savefig (name+'.jpg')
	plt.close()
	return 


def plot_percentage1 (x_label, y_label, title, name,   data):

	#bins = np.arange(0,np.max(arr)+1)
	#bins = np.arange(0,np.max(data)+1).astype(int)
	#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
	x= np.arange( 0, np.max(data)+1).astype(int)

	y= np.zeros_like(x)
	for idx in x:
		y[idx]=np.sum (data == idx)


	#pdb.set_trace()
	#ax= sns.violinplot (x=, y=y_label, data= data )
	
	#bins = len(np.unique(data))
	plt.hist(data, bins=5)
	#plt.bar (x=x, height=y, tick_label=list(x))
	#plt.xlabel(x_label)
	#plt.ylabel(y_label)
	#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
	plt.title(title)
	plt.savefig (name+'.jpg')
	plt.close()
	return 

def plot_percentage (x_label, y_label, title, name,   data):

	y , x = data	

	ax= sns.violinplot (x=x, y=y)
	
	#bins = len(np.unique(data))
	#plt.hist(data, bins=5)
	#plt.bar (x=x, height=y, tick_label=list(x))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
	plt.title(title)
	plt.savefig (name+'.jpg')
	plt.close()
	return 

def differential_analysis(x1_list, construct_list, title=None, case='case_1'):

	elements = []


	for i in np.unique (construct_list):

		#elements = x1_list [x1_list == 0]
	
		indices = (construct_list == i)
		new_list = x1_list [indices]
		

		if title is None :
			plot_histogram(y_label='Number of constructs in % (Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title='Histogram for number of Speaker-Turns for ' + class_dict[int(i)], name='utterance_'+case + class_dict[int(i)], data=new_list)
		else:
			plot_histogram(y_label='Number of constructs in %(Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title=title +' for '+ class_dict[int(i)], name='utterance_'+case + class_dict[int(i)], data=new_list)

		#elements.append(sum(indices))
	#pdb.set_trace()
	return elements

def subject_analysis (x1_list,construct_list, subject_list,version='same', case='case_1'):

	def plot_histogram (x_label, y_label, title, name,   data):

	#bins = np.arange(0,np.max(arr)+1)
	#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx) / len(data)
		
		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		plt.bar (x=x, height=y, tick_label=list(x))

		if len(x) > 20:	
			params = np.arange (min(x), np.max(x)+1, 5.0).astype(int)
			plt.xticks (ticks=params, labels=list(params), fontsize=14)	
		plt.xlabel(x_label, fontsize=15)
		plt.ylabel(y_label, fontsize=13)
		plt.ylim(0.0,0.8)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title, fontsize=13)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	d = {'1': 'Child', '2': 'Mother'}
	

	for jj in np.unique (construct_list):
		
		
		indexes= construct_list == jj 
		x_list =  x1_list[indexes]
		sub_list= subject_list[indexes]

		for i in np.unique (subject_list):
			

			indices = sub_list == i
			new_list = x_list [indices]

			if version == 'same':
				plot_histogram (y_label='Number of constructs in %(Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title='Histogram for number of Speaker-Turns for ' + d[i] +'-'+class_dict[int(jj)], name='utterance_'+ case + d[i] + class_dict[int(jj)], data=new_list)
			else:
				plot_histogram (y_label='Number of constructs in %(Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title='Histogram for number of Other Speaker turns for ' + d[i] +'-'+ class_dict[int(jj)], name='utterance_'+ case + d[i]+ class_dict[int(jj)], data=new_list)
	
	return 

def subject_percentage1 (x1_list,construct_list, subject_list,version='same', case='case_1'):


	d = {'1': 'Child', '2': 'Mother'}

	for jj in np.unique (construct_list):

		indexes= construct_list == jj 
		x_list =  x1_list[indexes]
		sub_list= subject_list[indexes]

		for i in np.unique (subject_list):
			

			indices = sub_list == i
			new_list = x_list [indices]

			if version == 'same':
				plot_percentage (y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Proportion of Turn time/construct', title='Histogram for number of Speaker Time  for ' + d[i] +'_'+class_dict[int(jj)], name='utterance_'+ case + d[i] + class_dict[int(jj)], data=new_list)
			else:
				plot_percentage (y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Proportion of Turn time/construct', title='Histogram for number of Other Speaker Time for ' + d[i] +'_'+ class_dict[int(jj)], name='utterance_'+ case + d[i]+ class_dict[int(jj)], data=new_list)
	
	return 


def violin_plot (x1_list,construct_list, subject_list,version='same', case='case_1'):

	def plot_percentage (x_label, y_label, title, name,   data):

		y , x = data	

		ax= sns.violinplot (x=x, y=y)
		
		#bins = len(np.unique(data))
		#plt.hist(data, bins=5)
		#plt.bar (x=x, height=y, tick_label=list(x))
		plt.xlabel(x_label)
		plt.ylabel(y_label, fontsize=12)
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title)
		plt.savefig (name+'.jpg')
		plt.close()
		return 
	d = {'1': 'Child', '2': 'Mother'}

	
	for i in np.unique (subject_list):
			

			indices = subject_list == i
			new_list = x1_list [indices]
			new_construct_list= construct_list[indices]

			sorted_indices = np.argsort(new_construct_list)
			new_list = new_list [sorted_indices]
			new_construct_list= new_construct_list[sorted_indices]
			love_list = np.array([class_dict[int(j)] for j in new_construct_list])
			print (min(new_list))
			if version == 'same':
				plot_percentage (y_label='Proportion of Turn Time / Construct', x_label='Constructs', title='Violin Plot of Same Speaker in '+ d[i]+' constructs', name='utterance_'+ case + d[i] , data=[new_list, love_list])
			else:
				plot_percentage (y_label='Proportion of Turn Time / Construct', x_label='Constructs', title='Violin Plot of Other Speaker in' + d[i]+' constructs' , name='utterance_'+ case + d[i], data=[new_list,love_list])
	
	return 


def construct_per_turn (t_start,t_end,\
							p_start_time, p_end_time, \
							c_start_time, c_end_time, \
							parent_speaker, child_speaker, \
							t_speaker, t_duration , \
							parent_duration, child_duration, \
							parent_label, child_label,\
							table_dict=None, \
							duration_dict=None):


	arr1= np.zeros_like(t_start)

	other_speaker_arr1 = np.zeros_like(t_start)


	for idx, (start_t , end_t , speaker_t, duration) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

		if speaker_t == '1':
			for c_idx , (start_c, end_c, label_c ) in enumerate (zip (c_start_time, c_end_time, child_label)):

				if (start_c < start_t  and  end_c > start_t and end_c < end_t ) or \
				   (start_c < end_t  and end_c > end_t and start_t <start_c) or \
				   (start_c >= start_t and end_c <= end_t) or \
				   (start_c < start_t and end_c>end_t) :

					arr1[idx] += 1
				  	#speaker_label.append (int(label_c))

		elif speaker_t =='2':
			for p_idx , (start_p, end_p, label_p ) in enumerate (zip (p_start_time, p_end_time, parent_label)):

				if (start_p < start_t  and  end_p > start_t and end_p < end_t ) or \
				   (start_p < end_t  and end_p > end_t and start_t <start_p) or \
				   (start_p >= start_t and end_p <= end_t) or \
				   (start_p < start_t and end_p>end_t) :
					
					arr1[idx] += 1

					#speaker_label.append(int(label_p))

	#------------Other speaker construct -------------------------#
	for idx, (start_t , end_t , speaker_t, duration) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

		if speaker_t == '2':
			for c_idx , (start_c, end_c, label_c ) in enumerate (zip (c_start_time, c_end_time, child_label)):

				if (start_c < start_t  and  end_c > start_t and end_c < end_t ) or \
				   (start_c < end_t  and end_c > end_t and start_t <start_c) or \
				   (start_c >= start_t and end_c <= end_t) or \
				   (start_c < start_t and end_c>end_t) :

					other_speaker_arr1[idx] += 1
				  	#speaker_label.append (int(label_c))

		elif speaker_t =='1':
			for p_idx , (start_p, end_p, label_p ) in enumerate (zip (p_start_time, p_end_time, parent_label)):

				if (start_p < start_t  and  end_p > start_t and end_p < end_t ) or \
				   (start_p < end_t  and end_p > end_t and start_t <start_p) or \
				   (start_p >= start_t and end_p <= end_t) or \
				   (start_p < start_t and end_p>end_t) :
					
					other_speaker_arr1[idx] += 1
		

	return arr1 , other_speaker_arr1
def turn_pauses_overlaps (t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration,c_duration, label=None, text=None):
	
	overlap_count = np.zeros_like(t_start)
	overlap_duration= np.zeros_like(t_start).astype('float')

	pause_duration = np.zeros_like (t_start).astype('float')

	within_duration = np.zeros_like (t_start).astype('float')
	between_duration= np.zeros_like (t_start).astype('float')

	prev_end = 0
	prev_speaker= None

	
	for idx, (start_t , end_t , speaker_t, duration , turn) in enumerate(zip(t_start, t_end, t_speaker, t_duration, text)):


		# for overlaps
		if start_t < prev_end and idx > 0:
			overlap_count [idx] += 1
			overlap_duration[idx] = prev_end - start_t
		
		# for pauses
		if start_t >= prev_end and idx > 0:
			pause_duration[idx] = start_t - prev_end


		# For within speaker pauses 
		if start_t >= prev_end and speaker_t == prev_speaker and idx > 0:
			within_duration[idx] = start_t - prev_end

		# For between speaker pauses
		if start_t >= prev_end and speaker_t != prev_speaker and idx > 0:
			between_duration[idx]= start_t - prev_end


		prev_end = end_t 
		prev_speaker= speaker_t
		


	#overlap_count = np.roll(overlap_count,-1)
	#overlap_duration= np.roll(overlap_duration,-1)
	#pause_duration= np.roll(pause_duration,-1)
	#within_duration= np.roll(within_duration,-1)
	#between_duration= np.roll(between_duration,-1)
	
	within_indices = within_duration > 0 
	between_indices = between_duration > 0
	overlap_indices = overlap_count == 1
	#within_indices =0 
	#between_indices=0
	
	return [overlap_count, overlap_duration], [pause_duration, within_duration, between_duration], [within_indices, between_indices]

def turn_statistics (data_lists):

	turn_duration, overlap_count, overlap_duration, pause_duration, within_duration, between_duration, speaker= data_lists


	total_turns = speaker.size
	parent_turns = np.sum (speaker =='2')
	child_turns = np.sum (speaker =='1')
	
	child_duration = turn_duration[ speaker=='1']
	parent_duration = turn_duration [ speaker =='2']

	total_overlap = np.sum (overlap_count) 
	parent_overlap = np.sum(overlap_count [speaker == '2']) 
	child_overlap = np.sum (overlap_count [speaker == '1']) 

	
	total_ov_duration=  overlap_duration[overlap_count==1]
	child_ov_duration = overlap_duration [(speaker == '1') & (overlap_count==1)]
	parent_ov_duration= overlap_duration [(speaker == '2') & (overlap_count==1)]


	total_pause_duration = pause_duration [overlap_count==0]
	child_pause_duration = pause_duration [(speaker=='1') & (overlap_count==0)]
	parent_pause_duration = pause_duration [(speaker=='2') & (overlap_count==0)]


	total_within_duration = within_duration[within_duration > 0]
	child_within_duration = within_duration [ (speaker=='1') & (within_duration>0)]
	parent_within_duration= within_duration [ (speaker=='2') & (within_duration>0)]
	
	
	total_between_duration = between_duration[between_duration> 0]
	child_between_duration = between_duration [ (speaker=='1') & (between_duration>0)]
	parent_between_duration= between_duration [ (speaker=='2') & (between_duration>0)]
	

	print ("Total turns=", total_turns, "parent_turns=", parent_turns, float(parent_turns)/total_turns ,"child_turns=", child_turns, float(child_turns)/total_turns)

	
	print ("Total Overlap=", total_overlap, "parent_overlap=", parent_overlap, float(parent_overlap)/child_overlap, "child_overlap=", child_overlap,float(child_overlap)/total_overlap)

	print ("Turn duration=",np.mean (turn_duration), np.std (turn_duration), np.median(turn_duration))
	print ("child_duration=", np.mean(child_duration), np.std (child_duration), np.median(child_duration))
	print ("parent_duration=", np.mean(parent_duration), np.std(parent_duration), np.median(parent_duration))


	print ("Mean overlap duration", np.mean(total_ov_duration), "Standard Deviation", np.std(total_ov_duration),"median=", np.median(total_ov_duration))
	print ("Child Overlap Duration", np.mean(child_ov_duration), np.std(child_ov_duration), np.median(child_ov_duration))
	print ("Parent Overlap Duration", np.mean(parent_ov_duration), np.std(parent_ov_duration), np.median(parent_ov_duration))


	print ("Mean pause duration", np.mean(total_pause_duration), np.std(total_pause_duration), len(total_pause_duration), np.median(total_pause_duration))
	print ("Child pause_duration", np.mean(child_pause_duration), np.std(child_pause_duration), len(child_pause_duration), np.median(child_pause_duration))
	print ("Mother pause_duration", np.mean(parent_pause_duration), np.std(parent_pause_duration), len(parent_pause_duration), np.median(parent_pause_duration))


	print ("Mean within duration", np.mean(total_within_duration) , np.std (total_within_duration), len(total_within_duration), np.median(total_within_duration))
	print ("Child within duration", np.mean(child_within_duration), np.std (child_within_duration), len(child_within_duration), np.median(child_within_duration))
	print ("Parentt within duration", np.mean(parent_within_duration), np.std(parent_within_duration), len(parent_within_duration), np.median(parent_within_duration))

	print ("Mean between durationn", np.mean(total_between_duration), np.std (total_between_duration), len(total_between_duration), np.median(total_between_duration))
	print ("Child between duration", np.mean(child_between_duration), np.std (child_between_duration), len(child_between_duration), np.median(child_between_duration))
	print ("Parent between_duration", np.mean(parent_between_duration), np.std (parent_between_duration), len(parent_between_duration), np.median(parent_between_duration))

	pdb.set_trace()
	return


def plot_turns (x_list, subject_list, version='same', case='case1'):
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx)  / len(data)

		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		
		plt.bar (x=x, height=y, tick_label=list(x))
		#plt.locator_params(axis='x', nbins=8)
		
		if len(x) > 20:	
			params = np.arange (min(x), np.max(x)+1, 5.0).astype(int)
			plt.xticks (ticks=params, labels=list(params), fontsize=14)
		plt.xlabel(x_label, fontsize=16)
		plt.ylabel(y_label, fontsize=14)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		plt.ylim(0.0,0.8)
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title,fontsize=14)
		plt.savefig (name+'.jpg')
		plt.close()
		return 

	indices  = x_list  != 0 
	x_list = x_list [indices]
	subject_list = subject_list [indices]

	dict_speaker = {'1': 'Child', '2': 'Mother'}

    #------------General---------------------#
	if version == 'same':
		#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(x_list))+')', title="Histogram of Number of  all speaker constructs", name='turn_'+version +'_'+case, data=x_list )
		plot_histogram ("Number OF Construct/ Turn", " Number of Turns (in percentage)", title="Histogram of Number of  Same speaker constructs", name='turn_'+version +'_'+case, data=x_list )
	
	else:
		#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(x_list))+')', title="Histogram of Number of Other speaker constructs", name='turn_'+version +'_'+case, data=x_list )
		plot_histogram ("Number OF Construct/ Turn", " Number of Turns (in percentage)" , title="Histogram of Number of Other speaker constructs", name='turn_'+version +'_'+case, data=x_list )

	#-------Differential---------------#
	for i in np.unique(subject_list):
		
		sub_indices = subject_list == i 
		sub_list   = x_list [ sub_indices]
		
	
		if version == 'same':
			
			#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(sub_list))+')', title="Histogram of Number of all speaker  constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] + '_'+case, data=sub_list )
			plot_histogram ("Number OF Construct/ Turn", " Number of Turns (in percentage)", title="Histogram of Number of Same speaker  constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] + '_'+case, data=sub_list )

		else:
			#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(sub_list))+')', title="Histogram of Number of Other speaker constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] +'_'+ case, data=sub_list )
			plot_histogram ("Number OF Construct/ Turn", " Number of Turns (in percentage)", title="Histogram of Number of Other speaker constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] +'_'+ case, data=sub_list )

	return 
def plot_turns_strategy (x_list, subject_list, strategy_list, version='same', strategy=0):
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx)  / len(data)

		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		
		plt.bar (x=x, height=y, tick_label=list(x))
		#plt.locator_params(axis='x', nbins=8)
		
		if len(x) > 20:	
			params = np.arange (min(x), np.max(x)+1, 5.0).astype(int)
			plt.xticks (ticks=params, labels=list(params), fontsize=14)
		plt.xlabel(x_label, fontsize=16)
		plt.ylabel(y_label, fontsize=14)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		plt.ylim(0.0,0.8)
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title,fontsize=14)
		plt.savefig (name+'.jpg')
		plt.close()
		return 

	dict_strategy = {0:'Within-Turns', 1:'end-of-turn', 2:'Overlapping' }

	indices  = (x_list  != 0 ) & (strategy_list==strategy)
	x_list = x_list [indices]
	subject_list = subject_list [indices]

	dict_speaker = {'1': 'Child', '2': 'Mother'}

    #------------General---------------------#
	if version == 'same':
		#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(x_list))+')', title="Histogram of Number of  all speaker constructs", name='turn_'+version +'_'+case, data=x_list )
		plot_histogram ("Number OF Construct/ Segment", " Number of Segments (in percentage)", title="Histogram of Number of  Same speaker constructs", name='turn_'+version +'_'+dict_strategy[strategy], data=x_list )
	
	else:
		#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(x_list))+')', title="Histogram of Number of Other speaker constructs", name='turn_'+version +'_'+case, data=x_list )
		plot_histogram ("Number OF Construct/ Segment", " Number of Segments (in percentage)" , title="Histogram of Number of Other speaker constructs", name='turn_'+version +'_'+dict_strategy[strategy], data=x_list )

	#-------Differential---------------#
	for i in np.unique(subject_list):
		
		sub_indices = subject_list == i 
		sub_list   = x_list [ sub_indices]
		
	
		if version == 'same':
			
			#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(sub_list))+')', title="Histogram of Number of all speaker  constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] + '_'+case, data=sub_list )
			plot_histogram ("Number OF Construct/ Segment", " Number of Segments (in percentage)", title="Histogram of Number of Same speaker  constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] + '_'+dict_strategy[strategy], data=sub_list )

		else:
			#plot_histogram ("Number OF Construct/ Turn", " Number of Turns in % (Total="+str(len(sub_list))+')', title="Histogram of Number of Other speaker constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] +'_'+ case, data=sub_list )
			plot_histogram ("Number OF Construct/ Segment", " Number of Segments (in percentage)", title="Histogram of Number of Other speaker constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] +'_'+ dict_strategy[strategy], data=sub_list )

	return 


def plot_turn_length (x_list, subject_list, case='case1'):
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx)  / len(data)

		"Ffor perceentage"
		weights=np.ones(len(data)) / len(data)	
		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		#plt.bar (x=x, height=y, tick_label=list(x))
		plt.hist(data, weights= weights,bins=30)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		
		plt.title(title)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	dict_speaker = {'1': 'Child', '2': 'Mother'}


	#-------Differential---------------#
	for i in np.unique(subject_list):
		
		sub_indices = subject_list == i 
		sub_list   = x_list [ sub_indices]
		
	
		plot_histogram ("Turn duration (in sec)",  " Number of Turns (in percentage)", title="Distribution of "+dict_speaker[i]+" turn duration time", name='duration_'+ dict_speaker[i] + '_'+case, data=sub_list )
		
	plot_histogram ("Turn duration (in sec)", "Number of Turns (in percentage)", title="Distribution of turn duration", name='duration_'+case, data=x_list )
	
	return 

def plot_pause_length (x_list, subject_list, title='within-speaker', case='case1'):
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx)  / len(data)

		"Ffor perceentage"
		weights=np.ones(len(data)) / len(data)	
		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		#plt.bar (x=x, height=y, tick_label=list(x))
		plt.hist(data, weights= weights,bins=50)
		plt.xlabel(x_label, fontsize=14)
		plt.ylabel(y_label, fontsize=12)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		
		plt.title(title, fontsize=14)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	dict_speaker = {'1': 'Child', '2': 'Mother'}


	#-------Differential---------------#
	for i in np.unique(subject_list):
		
		sub_indices = subject_list == i 
		sub_list   = x_list [ sub_indices]
		
		x_title = ' '.join(("Distribution of",title,"pause duration time for",dict_speaker[i]))
		x_name  = ''.join((title,"_",dict_speaker[i],"_",case))
	
		plot_histogram ("Turn duration (in sec)",  " Number of Turns (in percentage)", title=x_title, name=x_name, data=sub_list )
	
	x_title = ' '.join(("Distribution of",title,"pause duration time"))
	x_name  = ''.join((title,"_",case))
	plot_histogram ("Turn duration (in sec)", "Number of Turns (in percentage)", title=x_title, name=x_name, data=x_list )
	
	return
def length_analysis (x1_list,construct_list, duration_list,  subject_list,version='same', case='case_1'):

	def plot_length (x_label, y_label, title, name,   data):

		y , x = data	


		plt.bar (x=x, height=y)
		#bins = len(np.unique(data))
		#plt.hist(data, bins=5)
		#plt.bar (x=x, height=y, tick_label=list(x))
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	d = {'1': 'Child', '2': 'Mother'}

	
	for i in np.unique (subject_list):
			

			indices = subject_list == i
			new_list = x1_list [indices]
			new_construct_list= construct_list[indices]
			new_duration_list = duration_list[indices]

			sorted_indices = np.argsort(new_duration_list)
			new_list = new_list [sorted_indices] + 1
			new_construct_list= new_construct_list[sorted_indices]
			new_duration_list= new_duration_list[sorted_indices]
			#love_list = np.array([class_dict[int(j)] for j in new_construct_list])
			print (max(new_duration_list))
			if version == 'same':
				plot_length (y_label='Number of Speaker-Turns', x_label='Construct duration (in sec)', title='Turn count  of Same Speaker in '+ d[i]+' constructs', name='utterance_'+ case + d[i] , data=[new_list, new_duration_list])
			else:
				plot_length (y_label='Number of Speaker-Turns', x_label='Construct duration (in sec)', title='Turn count  of Other Speaker in' + d[i]+' constructs' , name='utterance_'+ case + d[i], data=[new_list,new_duration_list])
	
	return 


def plot_codes (codes, indices, speaker, case='within-speaker'):
	class_dict={0:'None',1:'Agg', 2:'Dys', 3:'Pos', 4:'Other'}
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		#x= np.arange( 0, np.max(data)+1).astype(int)
		
		x= list (class_dict.keys())
		#pdb.set_trace()
		
		y= np.zeros_like(x).astype('float')
		
		for idx, d in enumerate(data):
			y[idx]= float(d) / sum(data)
		
		
		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		plt.bar (x=x, height=y, tick_label=list(class_dict.values()))
		plt.xlabel(x_label, fontsize=16)
		plt.ylabel(y_label, fontsize=12)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		plt.ylim(0.0,0.6)
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title,fontsize=14)
		plt.savefig (name+'.jpg')
		plt.close()
		return 

	count= np.zeros((5))
	codes = codes [indices]
	speaker = speaker[indices]

	

	for idx, c_item in enumerate(codes):


		if sum (c_item)  == 0 :
			count[0] +=1 

		count [1] += c_item[0]
		count [2] += c_item[1]
		count [3] += c_item[2]
		count [4] += c_item[3]
	
	title = ' '.join(("Number of constructs in",case,"turns"))
	y_label= "Number of constructs in %"
	x_label= "Construct categories"
	name = ''.join((case))
	plot_histogram (x_label = x_label, y_label= y_label, title=title, name= name, data=count)

	return

def combine_turns (turn_data):
	#---------------This method combines the same speaker turns together to form a single turn -----------------------#
	[family, start_time, end_time, duration, subject, text, sequence] = turn_data

	turn_start_list=[]
	turn_end_list =[]
	turn_subject_list=[]
	turn_duration_list=[]
	turn_text_list=[]
	turn_sequence_list=[]

	for idx, (t_family, t_start_time, t_end_time, t_duration, t_subject, t_text, t_sequence) in enumerate (zip(family, start_time, end_time, duration, subject, text, sequence)):


		prev= None 
		new_sub = []
		new_start = []
		new_end = []
		
		#---------------Text appending -----------------#
		new_text= ""   # Calculates the new string 
		new_text_list=[] # Append the new string 

		#--------------Sequences -----------------#
		new_sequence = []
		new_sequence_list =[]
		for j_idx, sub in enumerate(t_subject):

			#-------Initial state -----------------#
			if prev == None:
				prev = t_subject[0] 
				new_sub. append (t_subject[j_idx])
				new_start.append (t_start_time[j_idx])
				
				new_text = new_text + t_text[j_idx]
				new_sequence = new_sequence + t_sequence[j_idx]
				
				continue 

			#-----------State when the previous turn is also of the same speaker ---------------#
			if sub == prev :
				#new_text.extend (t_text[j_idx])
				new_text = new_text +  '|' + t_text[j_idx]
				new_sequence += t_sequence[j_idx]
				prev = sub
				
				continue 
			else: #----------If there is a switch ---------------
				new_text_list.append (new_text)
				new_sequence_list.append (new_sequence)

				new_text = ""
				new_text += t_text[j_idx]

				new_sequence= []
				new_sequence += t_sequence[j_idx]

				new_sub.append (t_subject[j_idx])
				new_start.append (t_start_time[j_idx])
				new_end.append (t_end_time[j_idx - 1])
				prev=sub
	
		new_end.append (t_end_time[-1])
	
		new_start = np.array (new_start)
		new_end = np.array (new_end)
		new_duration = new_end - new_start

		new_text_list.append (new_text)
		new_sequence_list.append (new_sequence)
		
		turn_sequence_list.append(new_sequence_list)
		turn_text_list.append(new_text_list)
		turn_start_list.append(new_start)
		turn_end_list.append(new_end)
		turn_duration_list.append(new_duration)
		turn_subject_list.append(new_sub)
		#print (new_duration.shape, new_start.shape)

		assert len(new_duration) == len(new_start) == len(new_end) == len(new_sequence_list) == len(new_text_list)
	#pdb.set_trace()
	data={}
	data ['family'] = family
	data ['start_time'] = turn_start_list
	data ['end_time']  = turn_end_list
	data ['duration'] = turn_duration_list
	data ['subject'] = turn_subject_list
	data ['text']  = turn_text_list
	data['sequence'] = turn_sequence_list
	
	return data



def turns_with_silence (turn_data):
	#----------This method is only useful if we want to construuct segmentation with silences as well as overlaps------#
	[family, start_time, end_time, duration, subject, text, sequence] = turn_data

	turn_start_list=[]
	turn_end_list =[]
	turn_subject_list=[]
	turn_duration_list=[]
	turn_text_list=[]
	turn_sequence_list=[]

	for idx, (t_family, t_start_time, t_end_time, t_duration, t_subject, t_text, t_sequence) in enumerate (zip(family, start_time, end_time, duration, subject, text, sequence)):
		
	
		new_start_turn = []		
		new_end_turn =[]
		new_subject= []
		new_duration =[]
		
		new_text=[]
		new_sequence=[]
		new_id =[]

		for j_idx, sub in enumerate(t_subject):

			next_id = j_idx + 1
		
			if next_id != len(t_subject):
				#Overlap condition has to be at the first 
				if t_start_time [next_id] < t_end_time[j_idx] : #Overlap condition
					new_start_turn.append (t_start_time[j_idx])
					new_end_turn.append(t_start_time[next_id])
					new_duration.append(t_start_time[next_id]- t_start_time[j_idx])
					new_subject.append(t_subject[j_idx])
					new_id.append (j_idx)
					new_text.append (t_text[j_idx])
					new_sequence.append(t_sequence[j_idx])
					continue 
			#--- This tuurn is overlapping the previous turn and it's next turn is not overlapped
			if t_start_time[j_idx] < t_end_time[j_idx-1] and j_idx!=0: 
				
				#Insert the overlapped node  only if the present turn ends after the previous turn . 
				if t_end_time[j_idx] > t_end_time[j_idx-1]:
					new_start_turn.append (t_start_time[j_idx])
					new_end_turn.append (t_end_time[j_idx-1])
					new_subject.append("Both")
					new_duration.append (t_end_time[j_idx-1]- t_start_time[j_idx])
					new_id.append(j_idx-1)
					
					new_text.append (t_text[j_idx-1])
					new_sequence.append(t_sequence[j_idx-1])
				else: # This is an unique case when the previous turn is wrapping the current turn so it wiill end after the current turn 
					new_start_turn.append(t_start_time[j_idx])
					new_end_turn.append(t_end_time[j_idx])
					new_subject.append("Both")
					new_duration.append (t_end_time[j_idx] - t_start_time[j_idx])
					new_id.append(j_idx-1)
					new_text.append(t_text[j_idx-1])
					new_sequence.append(t_sequence[j_idx-1])
					
					#---------Now we enter the part of text of the previous turn which was left #
					new_start_turn.append(t_end_time[j_idx])
					new_end_turn.append (t_end_time[j_idx-1])
					new_subject.append(t_subject[j_idx-1])
					new_duration.append (t_end_time[j_idx-1] - t_end_time[j_idx])

					new_id.append (j_idx-1)
					new_text.append (t_text[j_idx-1])
					new_sequence.append(t_sequence[j_idx-1])


					#-----------Now we add another silence after the previos turn has been over --------
					if next_id != len(t_subject):
						if t_end_time[j_idx-1] < t_start_time[next_id]:
							new_start_turn.append (t_end_time[j_idx-1])
							new_end_turn.append(t_start_time[next_id])
							new_duration.append(t_start_time[next_id]- t_end_time[j_idx-1])
							new_subject.append('None')
							new_id.append(j_idx-1)
							new_text.append (" ")
							new_sequence.append([0])

					continue 				
				
				if next_id != len(t_subject):
					
					if t_end_time [j_idx] > t_start_time[next_id] :
						
						#----Now create the rest of the segment till the start of tthe next overlap
						new_start_turn.append (t_end_time[j_idx-1])
						new_end_turn.append (t_start_time[next_id])
						new_subject.append(t_subject[j_idx])
						new_duration.append (t_start_time[next_id]- t_end_time[j_idx-1])
						new_id.append(j_idx)
						
						new_text.append (t_text[j_idx])
						new_sequence.append(t_sequence[j_idx])
						continue 
				new_start_turn.append (t_end_time[j_idx-1])
				new_end_turn.append (t_end_time[j_idx])
				new_subject.append(t_subject[j_idx])
				new_duration.append (t_end_time[j_idx]- t_end_time[j_idx-1])
				new_id.append(j_idx)
			
				new_text.append (t_text[j_idx])
				new_sequence.append(t_sequence[j_idx])
			else:
				#Normal condition
				new_start_turn.append (t_start_time[j_idx])
				new_end_turn.append (t_end_time[j_idx])
				new_subject.append (t_subject[j_idx])
				new_duration.append (t_duration[j_idx])
				new_id.append(j_idx)
				new_text.append (t_text[j_idx])
				new_sequence.append(t_sequence[j_idx])
			#Silence condiition has to follow the normal condition to account for the gaps . Gaps are now where no speaker speaks. 
			if next_id != len(t_subject):
				if t_end_time[j_idx] < t_start_time[next_id]:
					new_start_turn.append (t_end_time[j_idx])
					new_end_turn.append(t_start_time[next_id])
					new_duration.append(t_start_time[next_id]- t_end_time[j_idx])
					new_subject.append('None')
					new_id.append(j_idx)
					new_text.append (" ")
					new_sequence.append([0])

	
		new_start_turn= np.array(new_start_turn)
		new_end_turn= np.array(new_end_turn)
		new_duration = np.array(new_duration)
		new_subject= np.array(new_subject)	
		new_id = np.array (new_id)
		
		turn_subject_list.append (new_subject)
		turn_start_list.append(new_start_turn)
		turn_end_list.append (new_end_turn)
		turn_duration_list.append(new_duration)
	
	#pdb.set_trace()
	data={}
	data ['family'] = family
	data ['start_time'] = turn_start_list
	data ['end_time']  = turn_end_list
	data ['duration'] = turn_duration_list
	data ['subject'] = turn_subject_list
	data ['text']  = turn_text_list
	data['sequence'] = turn_sequence_list
	
	return [family, turn_start_list, turn_end_list, turn_duration_list, turn_subject_list, turn_text_list, turn_sequence_list]

def create_constuct_table (t_start,t_end,\
							p_start_time, p_end_time, \
							c_start_time, c_end_time, \
							parent_speaker, child_speaker, \
							t_speaker, t_duration , \
							parent_duration, child_duration, \
							parent_label, child_label,\
							table_dict=None, \
							duration_dict=None):

	
	
	if table_dict == None  :	
		table_dict = {'None': np.zeros(16), '1': np.zeros(16), '2': np.zeros(16), 'Both': np.zeros(16)}
	if duration_dict == None  :	
		duration_dict = {'None': np.zeros(16), '1': np.zeros(16), '2': np.zeros(16), 'Both': np.zeros(16)}

	for idx, (start_t , end_t , speaker_t, duration) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):
		
		parent_meter =[]
		child_meter	=[]

		parent_overlap = []
		child_overlap = []
		for j_idx , (start_c, end_c, speaker_c, duration_c, label_c) in enumerate(zip(c_start_time,c_end_time,child_speaker, child_duration, child_label)):

			if (start_c < start_t  and  end_c > start_t and end_c < end_t ):
				child_meter.append(j_idx)
				child_overlap.append (end_c - start_t)
					
			elif (start_c < end_t  and end_c > end_t and start_t <start_c):
				child_meter.append(j_idx)
				child_overlap.append (end_c - end_t )

			elif (start_c >= start_t and end_c <= end_t):
				child_meter.append(j_idx)
				child_overlap.append(end_c - start_c)

			elif (start_c < start_t and end_c>end_t):
				child_meter.append(j_idx)
				child_overlap.append (end_t - start_t)

		for k_idx, (start_p, end_p, speaker_p, duration_p, label_p) in enumerate (zip(p_start_time, p_end_time, parent_speaker, parent_duration, parent_label)):
			if (start_p < start_t  and  end_p > start_t and end_t> end_p ):
				parent_meter.append(k_idx)
				parent_overlap.append (end_p - start_t)	
			elif (start_p < end_t  and end_p > end_t and start_t <start_p):
				parent_meter.append(k_idx)
				parent_overlap.append(end_p - end_t)
							
			elif (start_p >= start_t and end_p <= end_t):
				parent_meter.append(k_idx)
				parent_overlap.append(end_p - start_p)
				
			elif (start_p < start_t and end_p>end_t):
				parent_meter.append(k_idx)
				parent_overlap.append(end_t - start_t)

		eval_matrix = np.zeros((4,4))
		overlap_matrix = np.zeros((4,4))

		for ii, c_overlap in zip(child_meter, child_overlap):
			for jj, p_overlap in zip(parent_meter, parent_overlap):
				label_parent = int(parent_label[jj])
				label_child = int(child_label[ii])
				
				eval_matrix [label_child, label_parent] += 1
				overlap_matrix [label_child, label_parent] += float(c_overlap + p_overlap) / 2.0 
		
		feat =  eval_matrix.flatten()
		overlap_feat = overlap_matrix.flatten()

		table_dict[speaker_t] += feat
		duration_dict[speaker_t] += overlap_feat
	
	return table_dict, duration_dict


def get_construct_codes (t_start,t_end,\
							p_start_time, p_end_time, \
							c_start_time, c_end_time, \
							parent_speaker, child_speaker, \
							t_speaker, t_duration , \
							parent_duration, child_duration, \
							parent_label, child_label):



	speaker_label_list = []

	for idx , (start_t,  end_t, speaker_t ) in enumerate (zip (t_start, t_end, t_speaker)):
		speaker_label =[]

		if speaker_t == '1':
			for c_idx , (start_c, end_c, label_c ) in enumerate (zip (c_start_time, c_end_time, child_label)):

				if (start_c < start_t  and  end_c > start_t and end_c < end_t ) or \
				   (start_c < end_t  and end_c > end_t and start_t <start_c) or \
				   (start_c >= start_t and end_c <= end_t) or \
				   (start_c < start_t and end_c>end_t) :
				   speaker_label.append (int(label_c))

		elif speaker_t =='2':
			for p_idx , (start_p, end_p, label_p ) in enumerate (zip (p_start_time, p_end_time, parent_label)):

				if (start_p < start_t  and  end_p > start_t and end_p < end_t ) or \
				   (start_p < end_t  and end_p > end_t and start_t <start_p) or \
				   (start_p >= start_t and end_p <= end_t) or \
				   (start_p < start_t and end_p>end_t) :
				
				   speaker_label.append(int(label_p))

		
		if len(speaker_label) != 0:
			speaker_label_list.append (np.stack (speaker_label))
		else:
			#Since we always do not have information of constructs some speaker_label will be 0
			speaker_label_list.append(speaker_label)

	
	speaker_label_list = np.array(speaker_label_list)
	return  speaker_label_list

def create_constuct_transition (t_start,t_end,\
							p_start_time, p_end_time, \
							c_start_time, c_end_time, \
							parent_speaker, child_speaker, \
							t_speaker, t_duration , \
							parent_duration, child_duration, \
							parent_label, child_label, \
							construct_labels,\
							tables = None):


	

	if tables is None :
		total_transition = np.zeros ((4,4)).astype('float')
		total_count = np.zeros ((4,4)).astype('int')

		transition_tables = dict()
		count_tables = dict()
		
		parent_tables = dict ()
		parent_count_tables = dict()
		
		child_tables = dict()
		child_count_tables = dict()
		

		for i in range(total_transition.shape[1]):
			for j in range(total_transition.shape[1]):
				transition_tables[str(i)+str(j)] = []
				parent_tables[str(i)+str(j)] = []
				child_tables[str(i)+str(j)] = []
			
				child_count_tables[str(i)+str(j)] = 0
				parent_count_tables[str(i)+str(j)] = 0
				count_tables[str(i)+str(j)] = 0

		parent_transition = np.zeros ((4,4)).astype('float')
		parent_count = np.zeros ((4,4)).astype('int')
		
		child_transition = np.zeros ((4,4)).astype('float')
		child_count = np.zeros ((4,4)).astype('int')
	else:
		#[total_transition, total_count, parent_transition, parent_count, child_transition, child_count, transition_tables, parent_tables, child_tables] = tables	
	
		[transition_tables, count_tables,  parent_tables, parent_count_tables, child_tables, child_count_tables] = tables
	for idx , (start_t,  end_t, label_t ,speaker_t ) in enumerate (zip (t_start, t_end, construct_labels, t_speaker)):
	
		if idx == 0 :
			continue 		

		#----It means if there is a switch 

		if len(construct_labels[idx-1] ) != 0 and len(construct_labels[idx]) !=0 :  #Make sure that the transitions have legitimate labels 
			if t_start[idx]  > t_end [idx-1]:	

				transition_tables [str(construct_labels [idx-1][-1]) + str(construct_labels[idx][0])] .append (t_start[idx] - t_end [idx-1])
				count_tables [str(construct_labels [idx-1][-1]) + str(construct_labels[idx][0])]   += 1 
				
				'''
				total_transition[ construct_labels[idx-1][-1], construct_labels[idx][0]] += t_start[idx] - t_end[idx-1]		
				total_count[ construct_labels[idx-1][-1], construct_labels[idx][0]] += 1
				'''
			if t_start[idx]  > t_end [idx-1] and speaker_t == '1': #Transitioning from Parent to Child	
				parent_tables [str(construct_labels [idx-1][-1])+ str( construct_labels[idx][0])] .append (t_start[idx] - t_end [idx-1])
				parent_count_tables [str(construct_labels [idx-1][-1]) + str(construct_labels[idx][0])]   += 1 
				'''
				parent_transition[ construct_labels[idx-1][-1], construct_labels[idx][0]] += t_start[idx] - t_end[idx-1]		
				parent_count[ construct_labels[idx-1][-1], construct_labels[idx][0]] += 1
				'''
			if t_start[idx]  > t_end [idx-1] and speaker_t  =='2':	#Transitioning from child to parent
				child_tables [str(construct_labels [idx-1][-1])+str (construct_labels[idx][0])] .append (t_start[idx] - t_end [idx-1])
				child_count_tables [str(construct_labels [idx-1][-1]) + str(construct_labels[idx][0])]   += 1 
				'''
				child_transition[ construct_labels[idx-1][-1], construct_labels[idx][0]] += t_start[idx] - t_end[idx-1]		
				child_count[ construct_labels[idx-1][-1], construct_labels[idx][0]] += 1	
				'''


	

	#return  [total_transition, total_count, parent_transition, parent_count, child_transition, child_count, transition_tables, parent_tables, child_tables]
	return  [transition_tables, count_tables,  parent_tables, parent_count_tables, child_tables, child_count_tables]

def generate_transition_stats (table_info):
	[transition_tables, count_tables ] = table_info

	mean_table = dict()
	std_table = dict()

	total_table = dict()
	for idx, (key, value) in enumerate(transition_tables.items()):
		mean_table [key] = np.mean (value)
		std_table [key] = np.std (value)

	for idx, (key, value) in enumerate(count_tables.items()):
		total_table [key] = np.sum (value)

	return [mean_table, std_table, total_table]

def write_stats_to_csv (mean,std,total, name=None):
	col_name = {'00':'A-A', '01': 'A-D', '02':'A-P', '03':'A-O',\
				  '10': 'D-A', '11': 'D-D', '12':'D-P', '13':'D-O',\
				  '20':'P-A', '21':'P-D', '22':'P-P', '23':'P-O',\
				  '30':'O-A', '31':'O-D', '32':'O-P', '33':'0-0' }

	row_name = {0: 'Mean', 1:'Standard_Dev', 2:'Count'}
	import collections 

	if name is None:
		name = "table.xlsx"
	else:
		name = "transition_" + name + '.xlsx'

	#pdf = pd.DataFrame( mean)
	mean_d= collections.OrderedDict(sorted(mean.items()))
	std_d= collections.OrderedDict(sorted(std.items()))
	total_d= collections.OrderedDict(sorted(total.items()))
	
	pdf = pd.DataFrame.from_dict ([mean_d, std_d, total_d])
	pdf.rename (index=row_name, columns = col_name, inplace=True)

	with pd.ExcelWriter(name) as writer:
		pdf.to_excel(writer, sheet_name='sheet_1')
	return 


def write_dict_to_csv(data_dict, output_file=None):
	
	col_name = {0:'A-A', 1: 'A-D', 2:'A-P', 3:'A-0',\
				  4: 'D-A', 5: 'D-D', 6:'D-P', 7:'D-O',\
				  8:'P-A', 9:'P-D', 10:'P-P', 11:'P-O',\
				  12:'O-A', 13:'O-D', 14:'O-P', 15:'0-0' }
	row_name = {'1': 'Child', '2': 'Mother', 'Both':'Both', 'None':'Neither'}
	if output_file is None :
		output_file = 'table.xlsx'
	else:
		output_file= output_file + ".xlsx"

	pdf = pd.DataFrame (data_dict). T
	pdf.rename (index=row_name,columns= col_name, inplace=True)

	with pd.ExcelWriter(output_file) as writer:
		#writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
		pdf.to_excel (writer, sheet_name='Sheet1')
	return 


def turn_label_per_construct  (t_start,t_end,\
							p_start_time, p_end_time, \
							c_start_time, c_end_time, \
							parent_speaker, child_speaker, \
							t_speaker, t_duration , \
							parent_duration, child_duration, \
							parent_label, child_label
							):

	#-----------Give corresponding labels for segments ----- 

	# Places -1  if segment with construct
	turn_labels = np.empty (len(t_start))
	turn_onset = np.empty (len(t_start))
	turn_offset = np.empty (len(t_end))

	turn_labels.fill (-1)
	turn_onset.fill (-1)
	turn_offset.fill (-1)
	
	for c_idx , (start_c, end_c, label_c ) in enumerate (zip (c_start_time, c_end_time, child_label)):
		for idx , (start_t,  end_t, speaker_t ) in enumerate (zip (t_start, t_end, t_speaker)):

			if speaker_t == '1': #Child

				if (start_t >= start_c and end_t <= end_c) :
				
					turn_labels [idx] = label_c 
				
	for p_idx , (start_p, end_p, label_p ) in enumerate (zip (p_start_time, p_end_time, parent_label)):
		for idx , (start_t,  end_t, speaker_t ) in enumerate (zip (t_start, t_end, t_speaker)):

			if speaker_t == '2': #Parent

				if (start_t >= start_p and end_t <= end_p) :
				
					turn_labels[idx] = label_p


						
	return  turn_labels

def multi_seg_info (t_start, t_end , p_start_time, p_end_time, c_start_time, c_end_time, parent_speaker, child_speaker, t_speaker, t_duration, parent_duration, child_duration, parent_label, child_label):

	mask_list = np.zeros_like (t_start)
	multi_onset_list = []
	multi_offset_list = []
	multi_label_list = []

	flag = False
	for idx, (start_t , end_t , speaker_t, duration) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):
		onset_seg = []
		offset_seg = []
		label_seg = []
		
		if speaker_t == '1':	
			for c_idx , (start_c, end_c, label_c) in enumerate(zip(c_start_time,c_end_time, child_label)):

				if (start_c < start_t  and  end_c > start_t and end_c < end_t ):
					
					onset_seg.append (start_t)
					offset_seg.append (end_c)
					label_seg.append (label_c)
						
				elif (start_c < end_t  and end_c > end_t and start_t <start_c):
					onset_seg.append (start_c)
					offset_seg.append (end_t)
					label_seg.append (label_c)

				elif (start_c >= start_t and end_c <= end_t):
					onset_seg.append (start_c)
					offset_seg.append (end_c)
					label_seg.append (label_c)

				elif (start_c < start_t and end_c>end_t):
					onset_seg.append (start_t)
					offset_seg.append (end_t)
					label_seg.append (label_c)

		elif speaker_t == '2':
			for p_idx, (start_p, end_p,  label_p) in enumerate (zip(p_start_time, p_end_time, parent_label)):
				if (start_p < start_t  and  end_p > start_t and end_t> end_p ):
					onset_seg.append (start_t)
					offset_seg.append (end_p)
					label_seg.append (label_p)

				elif (start_p < end_t  and end_p > end_t and start_t <start_p):
					onset_seg.append (start_p)
					offset_seg.append (end_t)
					label_seg.append (label_p)

				elif (start_p >= start_t and end_p <= end_t):
					onset_seg.append (start_p)
					offset_seg.append (end_p)
					label_seg.append (label_p)

				elif (start_p < start_t and end_p>end_t):
					onset_seg.append (start_t)
					offset_seg.append (end_t)
					label_seg.append (label_p)


		if len (onset_seg) >= 1:
			mask_list [idx] = 1
		
		
		multi_onset_list.append (onset_seg)
		multi_offset_list.append (offset_seg)
		multi_label_list.append (label_seg)

	multi_onset_list = np.array ( multi_onset_list)
	multi_offset_list = np.array ( multi_offset_list)
	multi_label_list = np.array (multi_label_list)
	
	return mask_list, multi_onset_list, multi_offset_list, multi_label_list 


def get_turn_strategies (t_start, t_end, t_speaker):

	#--- O is the segment which will be overlapped 
	#----1 is the segment which is end-of-turn
	#----2 is the segment which is between turn 

	strategy = np.empty (len(t_start))
	strategy.fill (-1)

	prev_end = 0
	prev_speaker= None

	
	for idx, (start_t , end_t , speaker_t) in enumerate(zip(t_start, t_end, t_speaker)):

		#--- The first segment cannot be determined ----#
		if idx == 0 : 
			continue 


		#-------Switches or End of turns------#
		if (t_start [idx] >= t_end [idx-1]) and (t_speaker [idx] != t_speaker[idx-1]): 
			strategy [idx-1] = 1 

		#-------Holds or Within-Turns
		if (t_start [idx] >= t_end [idx-1]) and (t_speaker [idx] == t_speaker [idx-1]):
			strategy [idx-1] = 0


		if (t_start [idx] < t_end [idx-1]) :
			strategy[idx-1] = 2
		
	strategy [-1] = 1
	

	return  strategy


def get_continuous_labels (t_start, t_end, t_subject, target_start, target_end, target_window):
	speaker1_labels = np.zeros_like(target_start)
	speaker2_labels = np.zeros_like(target_end)

	for idx, (start, end) in enumerate(zip(target_start,target_end)):

		for t_idx, (st, ed) in enumerate(zip(t_start, t_end)):
			if (st <= start and ed >= start) or (st > start and st<end and ed > end ) or (st < start and ed>start and ed < end) or (st >=start and st<=end):
				if t_subject[t_idx] == '1':
					speaker1_labels[idx] = 1
				if t_subject[t_idx] == '2':
					speaker2_labels[idx] = 1



	'''
	start = 0
	end = len(speaker1_labels) - 1

	no_of_frames = int (1 * target_window/ 0.1 )
	c_labels =[]
	p_labels =[]

	while start + no_of_frames < end :
		
		c_labels.append (speaker1_labels[start:start+no_of_frames])
		p_labels.append (speaker2_labels[start:start+no_of_frames])
		start= start + 1

	c_labels= np.stack(c_labels)
	p_labels= np.stack(p_labels)
	pdb.set_trace()
	'''
	return  [speaker1_labels, speaker2_labels]

def read_words(root_dir, family):

	c_pd = pd.read_excel (os.path.join(root_dir, str(family)+'_1.xlsx' ))
	p_pd = pd.read_excel (os.path.join(root_dir, str(family)+'_2.xlsx' ))

	c_start, c_stop, c_duration, c_words = c_pd ['Start'].values , c_pd['Stop'].values,  c_pd['Duration'].values ,c_pd ['Transcription'].values
	p_start, p_stop, p_duration, p_words= p_pd ['Start'].values , p_pd['Stop'].values,  p_pd['Duration'].values, p_pd ['Transcription'].values

	c_speaker = ['1'] * len (c_start)
	p_speaker = ['2'] * len (p_start)
	speaker = np.array(c_speaker + p_speaker)
	start, stop, duration, words= np.concatenate ((c_start, p_start)), np.concatenate((c_stop, p_stop)),\
									np.concatenate ((c_duration, p_duration)), np.concatenate((c_words, p_words))

	sorted_indices = np.argsort(start)
	start = start[sorted_indices]
	stop = stop [sorted_indices]
	duration= duration [sorted_indices]
	words = words [sorted_indices] 
	speaker = speaker [sorted_indices]


	w_d =dict()
	w_d ['start'], w_d['stop'] = start, stop
	w_d ['duration'], w_d ['words'] = duration, words	
	w_d ['speaker'] = speaker
	return w_d
#def find_end_of_turns (t_start, t_end, t_subject, t_strategy):
def find_end_of_turns (speaker_1, speaker_2, start,  end, t_strategy, window_size=0.2, future_window=1.0):
	#window size and future window in seconds
	
	eval_labels= np.zeros_like(speaker_1)
	strategy_labels = np.zeros_like(speaker_1)
	
	f_frame = int(future_window / window_size)
	for idx, (sp1, sp2, st, end) in enumerate(zip(speaker_1, speaker_2, start,end)):	
		if sp1==0 and sp2 ==0:
			offset = idx+1  #Calculate the future window from the next frame (after 200ms has passed)
			#Case 1:Speaker 1 speaks and Speaker 2 does not speak 
			if speaker_1[offset: offset+ f_frame].any() and  not speaker_2[offset:offset+f_frame].any():
				eval_labels[idx] = 1
				
				if speaker_1[idx-1] == 1:
					strategy_labels[idx]= 1
				elif speaker_2[idx-1] == 1:
					strategy_labels[idx]= 2
				else:
					pass 
			#Case 2: Speaker 2 speaks and Speaker 1 does not speak
			elif speaker_2[offset: offset+ f_frame].any() and not speaker_1[offset:offset+ f_frame].any():
				eval_labels[idx] = 2
				
				if speaker_2[idx-1] == 1:
					strategy_labels[idx]= 1
				elif speaker_1[idx-1] == 1:
					strategy_labels[idx]= 2 
				else:
					pass 
			# Cases of overlaps and Pauses greater than future window are avoided

	eval_labels[strategy_labels == 0] = 0 #Change consecutive indicators to 0 s
	
	'''
	for idx, val in enumerate(eval_labels):
		if val != 0: 
			print ("idx=",idx, speaker_1[idx-2:idx+f_frame], speaker_2[idx-2:idx+f_frame], strategy_labels[idx])
			pdb.set_trace()
	'''
	return eval_labels, strategy_labels

def get_proportion (s_start, s_end, t_start, t_end):
	prop = 0.0
	if s_start >= t_end:
		return 0.0, False
	if s_start >= t_start and s_end <= t_end :
		prop = (s_end - s_start) / (t_end - t_start)
	elif s_start < t_start and s_end > t_end :
		prop = 1.0 
	
	elif s_start < t_start and s_end > t_start and s_end <= t_end:
		prop = (s_end  - t_start) / (t_end - t_start)
	elif s_start <  t_end  and s_end > t_end and s_start >= t_start:
		prop = (t_end - s_start) / (t_end - t_start)
	
	return prop, True
def get_linguistic (w_d, audio_start, audio_end, c_labels, p_labels):
	
	sp1_props_l = []
	sp2_props_l = []
	sp1_words_l = []
	sp2_words_l =[]

	for idx, (start, end) in enumerate (zip (audio_start, audio_end)):
		sp1_props, sp1_words = [], []
		sp1 ='1' if c_labels[idx] == 1 else '0' 
		indices = [False] * len(w_d['start'])
		for j_idx, (w_start, w_end) in enumerate(zip (w_d['start'], w_d['stop'])):
			prop, exist = get_proportion ( w_start, w_end , start, end )
	
			if exist and prop > 0.0:
				if w_d['speaker'][j_idx] == sp1:
					sp1_props.append (prop)
					sp1_words.append (w_d['words'][j_idx])
					indices[j_idx]=True
			elif not exist:
				break 
	
		sp1_props = np.array(sp1_props)
		sp1_words = np.array(sp1_words)
		if not sp1_props.any():
			sp1_props = np.array([1.0])
			sp1_words = np.array(['<blank>'])
		sp1_props_l.append (sp1_props)
		sp1_words_l.append (sp1_words)
		
		#print (start, end, sp1_props, sp1_words, w_d['start'][indices], w_d['stop'][indices])
		#pdb.set_trace()
	for idx, (start, end) in enumerate (zip (audio_start, audio_end)):
		sp2_props, sp2_words = [], []
		sp2 ='2' if p_labels[idx] == 1 else '0'
		indices = [False] * len(w_d['start'])
		for j_idx, (w_start, w_end) in enumerate(zip (w_d['start'], w_d['stop'])):
			prop, exist = get_proportion ( w_start, w_end , start, end )
			if exist and prop > 0.0:
				if w_d['speaker'][j_idx] == sp2 :
				
					sp2_props.append (prop)
					sp2_words.append (w_d['words'][j_idx])
					indices[j_idx]= True
			elif not exist:
				break

		sp2_props = np.array(sp2_props)
		sp2_words = np.array(sp2_words)
		if not sp2_props.any():
			sp2_props = np.array([1.0])
			sp2_words = np.array(['<blank>'])
		
		sp2_props_l.append (sp2_props)
		sp2_words_l.append (sp2_words)
		
	ling_child, ling_mother = dict (), dict()
	ling_child['props'] = np.array(sp1_props_l)
	ling_child['text'] = np.array(sp1_words_l)
	ling_mother['props'] = np.array(sp2_props_l)
	ling_mother['text'] = np.array(sp2_words_l)
	return ling_child, ling_mother

def analyze_for_data(res, prepare_data = False, multi_seg= False):
	#[family, start_time, end_time, duration, subject, text, sequence]= load_text_data()
	turn_data = load_text_data()
	#turn_data= turns_with_silence (turn_data)
	[family, start_time, end_time, duration, subject, text, sequence]= turn_data


	#-------------------------Read the ground truth_data --------------#s
	gr_data = np.load ('/home/sas479/tpot/data_analysis/ground_truth_data/phq_data.npy', allow_pickle=True, encoding='latin1').item()
	gr_fam , gr_group = gr_data['family_id'], gr_data['group']
	gr_fam = np.array (gr_fam)
	gr_group = np.array(gr_group)

	def convert_annot_to_code (annot):

		code_vec = np.zeros ((len(annot),4))

		for idx, annot_list in enumerate(annot) :
			if len(annot_list) != 0:
				for j in annot_list:
					code_vec[idx,j]+=1

		return code_vec

	def convert_speaker_to_code (speaker):


		code_vec = np.zeros((len(speaker),2))

		for idx, code in enumerate(speaker):
			if code  == '2':
				code_vec[idx,1]=1
			else:
				code_vec[idx,0]=1 
		return  code_vec 

	audio=res['audio']
	video=res['video']
	label=res['label']
	speaker=res['speaker']
	time=res['time']
	frame=res['frame']
	filename=res['filenames']
	p_time  = res['parent_time']
	c_time  = res['child_time']
	parent_speaker	= res ['parent_speaker']
	child_speaker = res ['child_speaker']
	parent_label = res['parent_label']
	child_label = res ['child_label']
	#turn_filename=res['turn_filename']
	#turn_speaker=res['turn_speaker']
	#turn_duration=res['turn_duration']
	#turn_time=res['turn_time'] 
	#turn_frame=res['turn_frame']


	
	construct_list=[]
	speaker_list = []

	neg1_list=[]
	neg2_list=[]

	speakert_list=[]
	turn_start_list=[]
	turn_end_list=[]
	


	turn_speaker_list = []
	turn_family_list=[]
	turn_eval_list, turn_strategy_list = [], []
	turn_audio_start_list, turn_audio_end_list = [],[]
	turn_video_start_list, turn_video_end_list = [],[]
	turn_text_child_list, turn_text_mother_list = [],[]
	turn_label_child_list, turn_label_mother_list = [], []


	#-------For multiple subsegments --------#

	multi_onset_list = []
	multi_offset_list = []
	multi_label_list = []

	onset = [x[0] for x in time]
	offset = [x[1] for x in time]

	p_start_time = [x[0] for x in p_time]
	p_end_time = [x[1] for x in p_time]
	c_start_time = [x[0] for x in c_time]
	c_end_time = [x[1] for x in c_time]


	file_list = [x[0] for x in filename]
	file_list = [x.split('_')[0][:-1][-4:] for x in file_list]
	
	state_table= None 
	duration_table = None
	table_info = None


	window_size=0.2
	future_window = 3.0 
	no_of_frames = 1 * future_window/ window_size

	c_labels = []
	p_labels = []

	def process (idx, fam ):

		check_fam =np.array( [x.find(fam) for x in file_list]) 	
		check_fam = np.where(check_fam>-1)[0]
			

		proc_data = dict()
		audio_start, audio_end = [],[]
		video_start, video_end = [],[]
		text_start, text_end = [],[]

		if len (check_fam) > 0:
			#get word data 
			
			w_d  = read_words (words_dir, fam )

			idx_fam = check_fam[0]

			value_1= onset [idx_fam]
			value_2= offset[idx_fam]

			#group_val = gr_group [ gr_fam == fam ] [0]
			begin = start_time[idx][0]
			end = end_time[idx][-1]

			count =0 
			while begin + window_size < end :
				audio_start.append(begin)

				audio_end.append(begin+window_size)
				video_start.append(begin)
				video_end.append(begin+window_size)
				begin = begin + window_size

			audio_start.append(begin)
			audio_end.append(end)
			video_start.append(begin)
			video_end.append(end)


			audio_start=np.array(audio_start)
			audio_end= np.array(audio_end)
			
			video_start= np.array(video_start)
			video_end= np.array(video_end)

			c_labels,p_labels=  get_continuous_labels (start_time[idx], end_time[idx], subject[idx], audio_start, audio_end, future_window)
			strategy = get_turn_strategies (start_time [idx], end_time [idx], subject[idx])

			eval_labels, strategy_labels=find_end_of_turns (c_labels, p_labels, audio_start, audio_end, strategy, window_size = window_size, future_window=1.0)
			
			ling_1, ling_2 = get_linguistic (w_d , audio_start, audio_end, c_labels, p_labels)

			

			proc_data ['family'] = fam 
			proc_data ['audio_start'] = audio_start
			proc_data ['audio_end'] = audio_end
			proc_data ['video_start'] = video_start
			proc_data ['video_end'] = video_end
			proc_data ['eval_labels'] = eval_labels
			proc_data ['strategy_labels'] = strategy_labels
			proc_data ['ling_child'] = ling_1
			proc_data ['ling_mother'] = ling_2 
			proc_data ['c_labels'] = c_labels
			proc_data ['p_labels'] = p_labels
			
			return proc_data
	
	proc_data_l = Parallel(n_jobs=1)(delayed(process)(idx,fam) for idx, fam in enumerate(family))
	

	
	for idx, dev in enumerate(proc_data_l):
		if dev is not None :
			turn_family_list.append (dev['family'])
			turn_audio_start_list.append( dev['audio_start'])
			turn_audio_end_list.append(dev['audio_end'])
			turn_video_start_list.append  (dev['video_start'])
			turn_eval_list.append (dev['eval_labels'])
			turn_strategy_list.append (dev['strategy_labels'])
			turn_video_end_list.append (dev['video_end'])	
			turn_text_child_list.append(dev['ling_child'])
			turn_text_mother_list.append (dev['ling_mother'])
			turn_label_child_list.append (dev['c_labels'])
			turn_label_mother_list.append (dev['p_labels'])
		
		
	res['family'] = np.array(turn_family_list)
	res['audio_start'] = np.array(turn_audio_start_list)
	res['audio_end'] = np.array(turn_audio_end_list)
	res['video_start'] = np.array(turn_video_start_list)
	res['video_end'] = np.array(turn_video_end_list)
	res['eval_labels'] = np.array(turn_eval_list)
	res['strategy_labels'] = np.array(turn_strategy_list)
	res['ling_child'] = np.array(turn_text_child_list)
	res['ling_mother'] = np.array(turn_text_mother_list)
	res['c_labels'] = np.array(turn_label_child_list)
	res['p_labels'] = np.array(turn_label_mother_list)
	
	pdb.set_trace()
	return 

def analyze_for_statistics(res, prepare_data = False, multi_seg= False):
	#[family, start_time, end_time, duration, subject, text, sequence]= load_text_data()
	turn_data = load_text_data()
	#turn_data= turns_with_silence (turn_data)
	[family, start_time, end_time, duration, subject, text, sequence]= turn_data

	for idx, sub in enumerate(subject):

		if len (sub) == 1 :
			print (idx, family[idx])
	
	#-------------------------Read the ground truth_data --------------#s
	gr_data = np.load ('/home/sas479/tpot/data_analysis/ground_truth_data/phq_data.npy', allow_pickle=True, encoding='latin1').item()
	gr_fam , gr_group = gr_data['family_id'], gr_data['group']
	gr_fam = np.array (gr_fam)
	gr_group = np.array(gr_group)

	def convert_annot_to_code (annot):

		code_vec = np.zeros ((len(annot),4))

		for idx, annot_list in enumerate(annot) :
			if len(annot_list) != 0:
				for j in annot_list:
					code_vec[idx,j]+=1

		return code_vec

	def convert_speaker_to_code (speaker):


		code_vec = np.zeros((len(speaker),2))

		for idx, code in enumerate(speaker):
			if code  == '2':
				code_vec[idx,1]=1
			else:
				code_vec[idx,0]=1 
		return  code_vec 

	audio=res['audio']
	video=res['video']
	label=res['label']
	speaker=res['speaker']
	time=res['time']
	frame=res['frame']
	filename=res['filenames']
	p_time  = res['parent_time']
	c_time  = res['child_time']
	parent_speaker	= res ['parent_speaker']
	child_speaker = res ['child_speaker']
	parent_label = res['parent_label']
	child_label = res ['child_label']
	#turn_filename=res['turn_filename']
	#turn_speaker=res['turn_speaker']
	#turn_duration=res['turn_duration']
	#turn_time=res['turn_time'] 
	#turn_frame=res['turn_frame']


	same_emo_list = []
	other_emo_list = []
	x2_list=[]
	x3_list=[]
	x4_list=[]
	x5_list=[]
	construct_list=[]
	speaker_list = []

	neg1_list=[]
	neg2_list=[]

	y1_list=[]
	y2_list=[]
	y3_list=[]
	ny1_list=[]
	ny2_list=[]
	speakert_list=[]
	turn_start_list=[]
	turn_end_list=[]
	


	turn_speaker_list = []
	turn_label_list=[]
	turn_strategy_list = []
	turn_onset_list=[]
	turn_offset_list = []
	turn_text_list=[]
	turn_duration_list=[]
	turn_sequence_list=[]
	turn_family_list=[]



	#-------For multiple subsegments --------#

	multi_onset_list = []
	multi_offset_list = []
	multi_label_list = []

	onset = [x[0] for x in time]
	offset = [x[1] for x in time]

	p_start_time = [x[0] for x in p_time]
	p_end_time = [x[1] for x in p_time]
	c_start_time = [x[0] for x in c_time]
	c_end_time = [x[1] for x in c_time]


	file_list = [x[0] for x in filename]
	file_list = [x.split('_')[0][:-1][-4:] for x in file_list]
	
	state_table= None 
	duration_table = None

	table_info = None
	for idx, fam in enumerate(family):

		check_fam =np.array( [x.find(fam) for x in file_list]) 	
		check_fam = np.where(check_fam>-1)[0]



		if len (check_fam) > 0:
			idx_fam = check_fam[0]

			value_1= onset [idx_fam]
			value_2= offset[idx_fam]

			#group_val = gr_group [ gr_fam == fam ] [0]

			same, other= construct_per_turn (start_time[idx], end_time[idx],\
												p_start_time[idx_fam],p_end_time[idx_fam],\
												c_start_time[idx_fam], c_end_time[idx_fam],\
												parent_speaker[idx_fam], child_speaker[idx_fam], \
												subject[idx], duration[idx],\
												p_end_time[idx_fam]-p_start_time[idx_fam], \
												c_end_time[idx_fam]-c_start_time[idx_fam], \
												parent_label[idx_fam], child_label[idx_fam],\
												)

			strategy = get_turn_strategies (start_time [idx], end_time [idx], subject[idx])

			'''	
			[overlap_count,overlap_duration],[pause_duration, within_duration, between_duration],[within_indices,between_indices]= turn_pauses_overlaps (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx], value_2-value_1,
								label= label[idx_fam],\
								text= text[idx])
			'''
			'''
			state_table, duration_table= create_constuct_table (start_time[idx], end_time[idx],\
												p_start_time[idx_fam],p_end_time[idx_fam],\
												c_start_time[idx_fam], c_end_time[idx_fam],\
												parent_speaker[idx_fam], child_speaker[idx_fam], \
												subject[idx], duration[idx],\
												p_end_time[idx_fam]-p_start_time[idx_fam], \
												c_end_time[idx_fam]-c_start_time[idx_fam], \
												parent_label[idx_fam], child_label[idx_fam],\
												table_dict = state_table, \
												duration_dict = duration_table
												)
			'''
			'''
			if group_val == 1  or group_val == 2:
				same_labels    =	 get_construct_codes (start_time[idx], end_time[idx],\
													p_start_time[idx_fam],p_end_time[idx_fam],\
													c_start_time[idx_fam], c_end_time[idx_fam],\
													parent_speaker[idx_fam], child_speaker[idx_fam], \
													subject[idx], duration[idx],\
													p_end_time[idx_fam]-p_start_time[idx_fam], \
													c_end_time[idx_fam]-c_start_time[idx_fam], \
													parent_label[idx_fam], child_label[idx_fam])

		
				table_info 	   =	 create_constuct_transition (start_time[idx], end_time[idx],\
													p_start_time[idx_fam],p_end_time[idx_fam],\
													c_start_time[idx_fam], c_end_time[idx_fam],\
													parent_speaker[idx_fam], child_speaker[idx_fam], \
													subject[idx], duration[idx],\
													p_end_time[idx_fam]-p_start_time[idx_fam], \
													c_end_time[idx_fam]-c_start_time[idx_fam], \
													parent_label[idx_fam], child_label[idx_fam], \
													construct_labels= same_labels, \
													tables= table_info)

													
														
							
			'''
			'''
			turn_labels = turn_label_per_construct (start_time[idx], end_time[idx],\
												p_start_time[idx_fam],p_end_time[idx_fam],\
												c_start_time[idx_fam], c_end_time[idx_fam],\
												parent_speaker[idx_fam], child_speaker[idx_fam], \
												subject[idx], duration[idx],\
												p_end_time[idx_fam]-p_start_time[idx_fam], \
												c_end_time[idx_fam]-c_start_time[idx_fam], \
												parent_label[idx_fam], child_label[idx_fam]
												)
			'''

			'''
			#--------------For Multi-label--------------#
			multi_mask, multi_onset, multi_offset, multi_label= multi_seg_info (start_time[idx], end_time[idx],\
												p_start_time[idx_fam],p_end_time[idx_fam],\
												c_start_time[idx_fam], c_end_time[idx_fam],\
												parent_speaker[idx_fam], child_speaker[idx_fam], \
												subject[idx], duration[idx],\
												p_end_time[idx_fam]-p_start_time[idx_fam], \
												c_end_time[idx_fam]-c_start_time[idx_fam], \
												parent_label[idx_fam], child_label[idx_fam]
												)
			strategy = get_turn_strategies (start_time [idx], end_time [idx], subject[idx])
			'''
			'''

			#---------------Criteria --------------------------#
			#indices = turn_labels != -1 
			if multi_seg:
				indices = multi_mask == 1

			#----------Uncomment if you don't want data preparation --------------------#

			subject_id = subject [idx] [indices]
			
			
			strategy_id = strategy [indices]
			turn_onset_id = start_time[idx][indices]
			turn_offset_id = end_time [idx] [indices]
			
			text_id = text[idx] [indices]
			sequence_id = np.array(sequence[idx])[indices]
			duration_id =duration[idx][indices]


			
			assert len(subject_id) == len(text_id) == len(strategy_id)
			
			if multi_seg :
				multi_onset_id , multi_offset_id, multi_label_id = multi_onset[indices], multi_offset[indices], multi_label[indices]
				assert len(subject_id) == len(text_id) == len(strategy_id) == len(multi_onset_id)
				multi_onset_list.append (multi_onset_id)
				multi_offset_list.append (multi_offset_id)
				multi_label_list.append (multi_label_id)
			else:
				turn_labels_id = turn_labels [indices]
				turn_label_list.append (turn_labels_id)
			'''
			turn_speaker_list.append (subject[idx])
			same_emo_list.append (same)
			other_emo_list.append (other)
			turn_strategy_list.append(strategy)
			'''
			turn_speaker_list.append (subject_id)
			turn_strategy_list.append (strategy_id)
			turn_onset_list.append (turn_onset_id)
			turn_offset_list.append (turn_offset_id)
			turn_text_list.append (text_id)
			turn_sequence_list.append (sequence_id)
			turn_duration_list.append (duration_id)
			turn_family_list.append (fam)

			'''
		
	#----Transition analysis ------#
	'''
	[transition_tables, count_tables,  parent_tables, parent_count_tables, child_tables, child_count_tables] = table_info
	t_mean, t_std, t_tot  = generate_transition_stats ([transition_tables, count_tables])
	p_mean, p_std, p_tot = generate_transition_stats ([parent_tables, parent_count_tables])
	c_mean, c_std, c_tot = generate_transition_stats ([child_tables, child_count_tables]) 

	write_stats_to_csv (t_mean, t_std, t_tot,"total")
	write_stats_to_csv (p_mean, p_std, p_tot,"parent")
	write_stats_to_csv (c_mean, c_std, c_tot,"child")
	
	turn_speaker_list = np.concatenate(turn_speaker_list)
	
	'''
	'''
	#----------Normalizing the list------------#
	for ii in np.unique (turn_speaker_list):
		state_table[ii] /=  sum(state_table[ii])
		

	pdb.set_trace()
	write_dict_to_csv (state_table, output_file = "count_normalize_table")
	write_dict_to_csv (duration_table, output_file = "duration_table")
	'''
	turn_speaker_list = np.concatenate(turn_speaker_list)
	same_emo_list = np.concatenate (same_emo_list)
	other_emo_list = np.concatenate (other_emo_list)
	turn_strategy_list = np.concatenate (turn_strategy_list)

	
	for o in [0,1,2]:
		plot_turns_strategy (same_emo_list, turn_speaker_list , turn_strategy_list, version='same' , strategy=o)
		plot_turns_strategy (other_emo_list, turn_speaker_list, turn_strategy_list,version='other', strategy=o)

	'''
	if prepare_data:
		if not multi_seg:
			return [turn_family_list, turn_speaker_list, turn_label_list, turn_strategy_list, turn_onset_list, turn_offset_list, turn_text_list, turn_sequence_list, turn_duration_list]
		else:
			return [turn_family_list, turn_speaker_list, turn_label_list, turn_strategy_list, turn_onset_list, turn_offset_list, turn_text_list, turn_sequence_list, turn_duration_list, \
					multi_onset_list, multi_offset_list, multi_label_list]
	'''
	return 

root_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_continuous_combined_data/'

train_fold=sorted([name for name in os.listdir(root_dir) if name.startswith('train')])
valid_fold=sorted([name for name in os.listdir(root_dir) if name.startswith('valid')])
test_fold=sorted([name for name in os.listdir(root_dir) if name.startswith('test')])

output_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/dataset_new/'


def combine_splits_for_analysis (train_info, valid_info, test_info):


	total_info = dict ()

	for idx, keys in enumerate(train_info.keys()):
		total_info [keys] = train_info [keys] + valid_info [keys] + test_info[keys]
		#print (type(train_info[keys]))

	return total_info
for idx, (train,valid,test) in enumerate(zip(train_fold,valid_fold,test_fold)):



	train_info = np.load (os.path.join(root_dir,train),allow_pickle=True).item()
	valid_info = np.load (os.path.join(root_dir,valid),allow_pickle=True).item()
	test_info = np.load (os.path.join(root_dir,test),allow_pickle=True).item()

	total_info = combine_splits_for_analysis (train_info, valid_info, test_info)
	
	#train_data=create_dataset(train_info)
	#valid_data=create_dataset(valid_info)
	#test_data=create_dataset(test_info)
	res = analyze_for_data (total_info, prepare_data=True, multi_seg=False)

	'''
	[turn_family_list, turn_speaker_list, turn_strategy_list, turn_onset_list, turn_offset_list, turn_text_list, turn_sequence_list, turn_duration_list, turn_gap_list] = res
	res_dict = {}
	res_dict['family'] = turn_family_list
	res_dict['speaker']= turn_speaker_list
	res_dict['strategy'] = turn_strategy_list
	res_dict['turn_onset']= turn_onset_list
	res_dict['turn_offset'] = turn_offset_list
	res_dict['text'] = turn_text_list
	res_dict['glove_token'] = turn_sequence_list
	res_dict['turn_duration'] = turn_duration_list
	res_dict['turn_gap'] = turn_gap_list
	'''
	#np.save (output_dir + 'train_data_'+str(idx)+'.npy', train_data )
	#np.save (output_dir + 'valid_data_'+str(idx)+'.npy',valid_data)
	#np.save (output_dir + 'test_data_'+str(idx)+'.npy',test_data)
	#np.save ('turn_data.npy', res)
	if idx == 0:
		break 

#d = Parallel(n_jobs=10)(delayed(par_fold)(idx,files) for idx, files in enumerate(folds))
print ('Done')
print ("Done")	
	