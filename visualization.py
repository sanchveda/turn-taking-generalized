import matplotlib.pyplot as plt
import os 
import numpy as np 
import pdb
import shutil
import seaborn as sns
import torch
from sklearn.metrics import f1_score,classification_report, roc_curve, auc, cohen_kappa_score, confusion_matrix


from pathlib import Path 

from sklearn import preprocessing
from evaluation_metrics import *
from matplotlib.ticker import FixedLocator, FixedFormatter

from itertools import cycle
import pdb 

class_dict={0:'Aggressive',1:'Dysphoric',2:'Positive'}#,3:'Other'}
class_dict= {0:'hap', 1:'sad', 2:'neu', 3:'ang', 4:'exc', 5:'fru' }
class_diic= {0: 'Not EOT ', 1:'EOT'}
#class_dict={0:'Aggressive',1:'Dysphoric',2:'Positive'}
#class_dict = {0:'0', 1:'1', 2:'2',3:'3',4:'4',5:'5',6:'6'}
def plot_loss_functions(read_data,write_filename,train=False):

	

	if len(read_data) > 1:
		train_data,test_data=read_data
		
		plt.xlabel('Number of Epochs')
		plt.ylabel('Loss')
		plt.title('Loss over epochs')
		plt.ylim(0.0,3.5)
		plt.plot(train_data, label='Traning Loss')
		plt.plot(test_data, label='Validation Loss')
		plt.legend(loc="upper left")
		plt.savefig (write_filename+'.png')

		plt.close()

		return

	'''
	data= np.load (read_filename,allow_pickle=True)


	if not train:
		print (data)
		plt.xlabel('Number of epochs')
		plt.ylabel('Validataion Loss')
		plt.title('Validataion Loss over Epochs')
		plt.plot(data,label='Validataion Loss')
		plt.legend(loc='upper left')
		plt.savefig(write_filename+'.png')
	else:
		
		plt.xlabel('Number of epochs')
		plt.ylabel('Training Loss')
		plt.title('Training Loss over Epochs')
		
		data=np.mean(data,axis=1)
		
		#data= data.flatten()
		

		plt.plot(data,label='Training Loss')
		plt.legend(loc="upper left")
		plt.savefig(write_filename+'.png')
	'''
	return 
def plot_roc_multiclass (y_true,y_score,save_dir):
	
	

	y_binarize= preprocessing.label_binarize(y_true, classes=[0,1])

	
	y_aucroc4= roc_auc_score (y_binarize, y_score, average='weighted')
	
	#y_binarize=[y_binarize1,y_binarize2,y_binarize3,y_binarize4]
	#y_aucroc= [y_aucroc1, y_aucroc2, y_aucroc3, y_aucroc4]
	#-------This is the same as doing the following steps. Howeve the following steps are needed for plotting ROC-----
	n_classes= y_binarize.shape[1]

	fpr=dict ()
	tpr=dict ()
	roc_auc= np.zeros(n_classes).astype('float')
	for i in range (n_classes):
		
		pdb.set_trace()
		fpr[i], tpr[i], _ = roc_curve(y_binarize[:,i], y_score[:,i])
		roc_auc[i] = auc (fpr[i], tpr[i])

	for i in range(n_classes):
		title= "ROC curve for " + class_dict[i] +'.png'
		savename=  save_dir + 'roc_' + class_dict[i]
		plt.figure()
		lw = 2
		plt.plot(fpr[i], tpr[i], color='darkorange',
		         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(title)
		plt.legend(loc="lower right")
		plt.savefig(savename)
		plt.close()

	
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red'])
	title='ROC curves'
	savename= save_dir + 'roc_' + 'all' + '.png'
	plt.figure()
	lw=2
	for i, color in zip(range(n_classes),colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(class_dict[i], roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.savefig(savename)
	plt.close()

	return 


def plot_confusion_matrix(confusion_mat,name,save_name,normalize=False):
	
	if normalize:
		confusion_mat=confusion_mat.astype('float')/confusion_mat.sum(axis=1) [:,np.newaxis]


	ax=sns.heatmap(confusion_mat, vmin=0.0, vmax=1.0,annot=True , center=0.0, annot_kws={"size":10})
	ax.set_title(name)

	plt.show()
	plt.savefig(save_name+".png",dpi=300)
	plt.close()

def plot_attention (attention_map,label,pred,name,save_name):
	
	labels=label*10 + pred
	labels=np.array(labels.astype('int')).reshape(1,1)
	
	ax=sns.heatmap(attention_map, yticklabels= labels, vmin=0.0, vmax=1.0, annot=False, center=0, annot_kws={"size":10})
	
	#ax.set_title(name)
	#ax.yaxis.set_major_locator(FixedLocator(label))
	plt.savefig(save_name+'.png', dpi=300)
	plt.close()

def handle_attention_plot(attention, test_duration, pred,label,save_info):

	batchsize=64
	[savename , fold] = save_info

	
	dir_name = savename + 'attention_' + str(fold) + '/'
	
	if os.path.isdir (dir_name):
		shutil.rmtree(dir_name)
	os.mkdir(dir_name)
		
	i=0
	while i < len(attention):
		batch = attention[i:i+64]
		batch = nconvert(batch)
		
		y_true= label[i:i+64]
		y_pred= pred[i:i+64]
		duration = test_duration[i:i+64]

		tp_indices= y_true == y_pred

		tp_attention = batch[tp_indices]
		ms_attention = batch[~tp_indices]
		tp_duration = duration[tp_indices]
		ms_duration = duration[~tp_indices]
		
		tp_true, tp_pred= y_true[tp_indices], y_pred[tp_indices]
		ms_true, ms_pred= y_true[~tp_indices], y_pred[~tp_indices]

		
		if tp_attention.size:
			a_true = tp_attention [:1, :tp_duration[0]]
			t_true, p_true = tp_true[0], tp_pred[0] 

			a_false = ms_attention[:1, :ms_duration[0]]
			t_false, p_false= ms_true[0], ms_pred[0]

			plot_attention (a_true, t_true, p_true, "",dir_name+'true_'+str(i) )
			plot_attention (a_false,t_false,p_false,"",dir_name+'false_'+str(i))

		i=i+64
	
		

	return
def nconvert(attention_map):

	r_att=np.empty(0)
	for attend in attention_map:
		
		r_att=np.vstack([r_att,attend.cpu().numpy()]) if r_att.size else attend.cpu().numpy()

	return r_att

def ck_convert (attention_map):

	r_att=np.empty(0)
	count=0
	for attend in attention_map:
		
		r_att=np.vstack([r_att,attend]) if r_att.size else attend
		print (count)
		count=count+1
	return r_att

def compute_roc_binary(y_test,y_score,save_dir):

	fpr,tpr,_= roc_curve(y_test,y_score)
	roc_auc  = auc (fpr,tpr)

	savename = save_dir + 'ROC Curve.jpg'
	plt.plot(fpr,tpr,color='darkorange',lw=2, label='ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], color='navy', lw=2 , linestyle='--')
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title("ROC curve")
	plt.legend(loc="lower right")
	plt.savefig(savename)
	plt.close()


def print_classification_report (report,save_dir):
	report_path = save_dir + '_report.txt'
	text_file = open (report_path, "w")
	text_file.write(report)
	text_file.close()


	return 
#[train_loss,test_loss,config_list]= np.load('loss_dan_full.npy',allow_pickle=True)
'''
category='onset_video'
save_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/lstm/lstm_'+ category +'_pruned_attention/plot/'
#save_dir = '/etc/Sanchayan/lstm/lstm_' + category + '/plot/'

result_dir= '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/lstm/lstm_' + category + '_pruned_attention/result/'
#result_dir = '/etc/Sanchayan/lstm/lstm_' + category + '/result/'
result_files= sorted([name for name in os.listdir(result_dir) if name.startswith('result')])

data_path= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/onset_video/"
test_list= np.array(sorted([name for name in os.listdir(data_path) if name.startswith('test')]))

'''
category='dan_full'
save_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/dan/'+ category +'/plot/'
#save_dir = '/etc/Sanchayan/lstm/lstm_' + category + '/plot/'

result_dir= '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/dan/' +  category+ '/result/'
#result_dir = '/etc/Sanchayan/lstm/lstm_' + category + '/result/'
result_files= sorted([name for name in os.listdir(result_dir) if name.startswith('result')])

data_path= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/dan_full/"
test_list= np.array(sorted([name for name in os.listdir(data_path) if name.startswith('test')]))



action ='general'
if action == 'loss_plot':
	
	for i, config in enumerate(config_list):

		dir_name = save_dir + config + '/'
		if os.path.isdir (dir_name):
			shutil.rmtree(dir_name)
		os.mkdir(dir_name)

		for idx in range (len(train_loss)):

			loss_train=train_loss[idx][i]
			loss_test =test_loss [idx][i]

			write_name= dir_name + 'fold_' + str(idx) 
			plot_loss_functions([loss_train,loss_test], write_name,train=True)
elif action == 'dialogue':
	root_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/'
	#result_files = [ x for x in os.listdir('.') if x.startswith('result')]
	result_files = [x for x in os.listdir (root_dir) if x.startswith('data_speaker') and x.endswith('.npy')]
	result_ll = []
	
 

	for idx, result_dir in enumerate (result_files):	
		save_dir = '_'.join (['plot']+result_dir.split('_')[1:]) + '/'
		
		Path(save_dir).mkdir(parents=True, exist_ok=True)


		files=os.listdir(result_dir)
		
		best_score = None 
		best_label, best_prediction, best_mask, best_confusion_matrix, best_classification_report, best_probs = None, None, None, None, None, None
		
		total_true = []
		total_pred = []
		total_score = []

		for idx, name in enumerate(files):

			#config_name  = name .split('x')[0]
			res = np.load(os.path.join(result_dir,name),allow_pickle=True).item()
			
			class_report, y_true, y_pred, y_f_score,y_probs = res['class_report'],\
															res['y_true'],\
															res['y_pred'],\
															res['f1_score'],\
															res['score']
			
			
			#f_score = f1_score (y_true, y_pred, average='weighted',sample_weight= y_mask)
			total_true.append (y_true)
			total_pred.append(y_pred)
			total_score.append (y_probs)

			
			#plot_loss_functions ([train_loss, valid_loss], save_dir+config_name, train=True)

		
		total_true= np.concatenate(total_true)
		total_pred= np.concatenate(total_pred)
		total_score = np.concatenate(total_score)

		class_report= classification_report (total_true, total_pred, digits= 4)
		c_mat = confusion_matrix (total_true, total_pred )
		plot_confusion_matrix (c_mat, 'confusion_matrix', save_dir+'_confusion', normalize=True)
		
		
		'''
		best_probs = np.array(best_probs)
		indices= y_mask == 1
		roc_label = best_label[indices]
		roc_pred  = best_prediction[indices]
		roc_probs = best_probs[indices]
		'''
		#plot_roc_multiclass (roc_label, roc_probs, save_dir )
		print_classification_report(class_report,save_dir)
		compute_roc_binary(total_true, total_score, save_dir)


else :
	root_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/'
	#result_files = [ x for x in os.listdir('.') if x.startswith('result')]
	result_files = [x for x in os.listdir (root_dir) if x.startswith('data_speaker') and x.endswith('.npy')]
	result_ll = []
	
	files1 = [x for x in result_files if x.startswith('data_speaker_1')]
	files2 = [x for x in result_files if x.startswith('data_speaker_2')]
	
	res1_list , res2_list = [], []
	res1= []
	res2= []
	for idx, (f1, f2) in enumerate(zip(files1, files2)): 
			
		if f1.split('.')[0][:-6] == files1[idx-1].split('.')[0][:-6] or idx ==0:
			res1.extend ([f1])
			res2.extend ([f2])
			
		else:
			res1_list.append (res1)
			res2_list.append (res2)
			res1=[]
			res2=[]
			res1.extend([f1])
			res2.extend([f2])
			#res1.append (f1)
			#res2.append (f2)
	res1_list.append (res1)
	res2_list.append (res2)
	

	for idx, (f1,f2) in enumerate( zip(res1_list, res2_list)):
		
		comb_id = f1[idx].split('.')[0].split('_')[3][:-4] if len(f1[idx].split('.')[0].split('_')) == 5 else 'all'
		save_dir= '_'.join(['plot',comb_id])+'/'

		#save_dir = '_'.join (['plot']+result_dir.split('_')[1:]) + '/'
		
		Path(save_dir).mkdir(parents=True, exist_ok=True)
		total_true= []
		total_pred=[]
		for j_idx, (l1, l2) in enumerate(zip(f1,f2)):
			dat1= np.load (os.path.join(root_dir, l1), allow_pickle=True).item()
			dat2= np.load (os.path.join(root_dir, l2), allow_pickle=True).item()
			
			preds1 = dat1['child_preds']
			preds2=  dat2['mother_preds']
			labels = dat1['eval_labels']
			strategy = dat2['strategy_labels']

			max1 = np.mean (preds1[:,:5], axis=1)
			max2 = np.mean (preds2[:,:5], axis=1)
			val_concat = np.concatenate ([max1.reshape(-1,1), max2.reshape(-1,1)], axis=1)
			pred_val = np.argmax (val_concat, axis=1)
		
			labels_ = labels -1 
			strategy = strategy-1
			pred_st=np.zeros_like (pred_val)
			for kk,val in enumerate(pred_val == labels_) :
				if val:
					pred_st[kk] = strategy[kk]
				else:
					pred_st[kk] = 1 if strategy[kk]==0 else 0 
			
			total_true.append (strategy)
			total_pred.append (pred_st)
		total_true= np.concatenate(total_true)
		total_pred= np.concatenate(total_pred)
		class_report= classification_report (total_true, total_pred, digits= 4)
		print_classification_report(class_report,save_dir)
		print (class_report)

