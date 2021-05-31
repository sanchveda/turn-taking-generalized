import numpy as np
np.random.seed(1234)
import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle
import os

from pathlib import Path 
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support


#from model_siamese import SiameseNet , MLP_Model, ContrastiveLoss
from dataloader_siamese import TPOT_trainer, TPOT_tester
from model_siamese import MLP_general, ContrastiveLoss  #, BERT_emotion_classifier
#from model_lmf import LMF, MaskedLoss
#from transformers import AdamW, get_linear_schedule_with_warmup

import json
from utilities import *


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_TPOT_loaders(domain, data,  fold=0,  batch_size=32, valid=0.1,  indices = None, num_workers=0, pin_memory=False, speaker=1):
    
    #[tr_indices, vl_indices, ts_indices] = indices
    tr_indices = os.path.join(data, 'train_'+ str(fold) +'.npy')
    vl_indices = os.path.join(data, 'valid_'+ str(fold) +'.npy')
    ts_indices = os.path.join(data, 'test_'+ str(fold) +'.npy')
    
    trainset = TPOT_trainer(full_data = tr_indices, select_indices = None, domain= domain, speaker=speaker)
    
    

    #train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              #sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)


    validset = TPOT_trainer(full_data = vl_indices, select_indices =None, domain= domain,scaler= None, train=False,speaker=speaker)
   
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              shuffle=False,
                              #sampler=valid_sampler,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    testset = TPOT_tester(full_data = [ts_indices,vl_indices], select_indices = None, domain= domain,scaler = None, train=False,speaker=speaker)

    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    

    
    #return train_loader, valid_loader, test_loader
    return train_loader, valid_loader , test_loader



def train_or_eval_model(model, loss_function, dataloader, epoch, device=None, optimizer=None, train=False, speaker=1):
    losses = 0.0
    batches = 0
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    
    for data in dataloader:
        if train:
            optimizer.zero_grad()  
        
        
        sp1_audio, sp1_video, sp2_audio, sp2_video, label, sp1_token, sp2_token, sp1_props, sp2_props ,mask =\
                [d.to(device) for d in data[:10]] if device is not None else data[:5]

        
        #sp1_token, sp2_token = data[5], data[6]
        #sp1_props, sp2_props = data[7], data[8]
        family = data[-1]         

        if speaker ==1 :
            output = model ([sp1_audio,sp2_audio], [sp1_video, sp2_video], [sp1_token, sp2_token], [sp1_props, sp2_props], mask)
        else:
            output = model ([sp2_audio,sp1_audio], [sp2_video, sp1_video], [sp2_token, sp1_token], [sp2_props, sp1_props], mask)
        #print (output1, output2)
        output_ = output.view(-1,1)
        label_ = label.view(-1,1)
        loss = loss_function (output_, label_)
        #log_prob =mo(del (audio)
    
        losses += loss.item()        
        
     
        
        if np.isnan (loss.item()):
            pdb.set_trace()
        #print (loss)
        if train:
            loss.backward()         
            optimizer.step()
    
        batches += 1 
    losses /= batches
    
    return losses#, [alphas, alphas_f, alphas_b, vids]
def test_model(model, loss_function, dataloader, epoch, device=None, optimizer=None, train=False, speaker=1):
    losses = 0.0
    batches = 0
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    
    family_list = []
    output_list = []
    eval_list = []
    strategy_list =[]
    for data in dataloader:
        if train:
            optimizer.zero_grad()  
        
        sp1_audio, sp1_video, sp2_audio, sp2_video, label, strategy, sp1_token, sp2_token, sp1_props, sp2_props ,mask =\
                [d.to(device) for d in data[:11]] if device is not None else data[:11]
        
        #sp1_token, sp2_token = data[5], data[6]
        #sp1_props, sp2_props = data[7], data[8]
        family = data[-1]         

        if speaker ==1 :
            output = model ([sp1_audio,sp2_audio], [sp1_video, sp2_video], [sp1_token, sp2_token], [sp1_props, sp2_props], mask)
        else:
            output = model ([sp2_audio,sp1_audio], [sp2_video, sp1_video], [sp2_token, sp1_token], [sp2_props, sp1_props], mask)
        #print (output1, output2)
        
        
        #print (loss)
        if train:
            loss.backward()         
            optimizer.step()
    
        family_list.extend (family)
        output_list.extend(output[:,-1,:].data.cpu().numpy())
        eval_list.extend(label.data.cpu().numpy())
        strategy_list.extend(strategy.data.cpu().numpy())
        batches += 1 
    losses /= batches
    family_list= np.array(family_list)
    output_list= np.array(output_list)
    eval_list= np.array(eval_list)
    strategy_list= np.array(strategy_list)

    return family_list, output_list, eval_list, strategy_list


def get_embeddings (model, dataloader, device=None):

    model.eval()
    speaker_stack=[]
    listener_stack=[]
    name_list =[]
    for data in dataloader:
        
        audio, video, l_audio, l_video, label = \
                [d.to(device) for d in data[:5]] if device is not None else data[:5]
        name = data[-1]

        output1, output2 = model (torch.cat((audio,video), dim=-1), torch.cat((l_audio,l_video), dim=-1))
    
        speaker_stack.extend (output1.cpu().data.numpy())
        listener_stack.extend (output2.cpu().data.numpy())
        
        name_list.extend (name)
    
    speaker_stack= np.array(speaker_stack)
    listener_stack=np.array(listener_stack)
    name_list = np.array(name_list)
    
    return speaker_stack, listener_stack, name_list

def make_dialog_data ( speaker_stack, listener_stack, name_list):
    res = dict()
    speaker_list = []
    listener_list = []
    names =[]
    for idx, name in enumerate(np.unique(name_list)):

        indices = name_list == name 
        speaker_dat = speaker_stack [indices]
        listener_dat = listener_stack [indices]
        speaker_list.append (speaker_dat)
        listener_list.append(listener_dat)
        names.append (name)
    res['speaker']= speaker_list
    res['listener']= listener_list
    return res 
def make_data ( speaker, preds, evals, strategy, name):
    res = dict()
    
    if speaker == 1:
        res['child_preds']= preds
    else:
        res['mother_preds']=  preds
    res['eval_labels']= evals
    res['strategy_labels']= strategy
    res['family']= name
    return res 

if __name__ == '__main__':
    
    fold_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/kfold_splits/'   
    data_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/data_general/'
    feature_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/'
    
    with open('wordmap.json') as fp:
        wordmap = json.load (fp)
       

    #full_data = combine_splits(data_dir)
    #feature_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/feature_both/'
    
    #full_data = np.load (os.path.join(data_dir,'FULL_DATA.npy'), allow_pickle=True).item()
    '''
    "Uncomment and use this only once . It will help in faster debugging in training. Otherwise the loading is computationally extensively"
    #Crreate fold data 
    for f_id in range(10):
        tr_indices , vl_indices , ts_indices = get_split_indices ( fold_dir = fold_dir, full_data = full_data, fold_no = f_id)



        data = dict ()
        for keys, values in full_data.items():
            data[keys] = np.array( values)[tr_indices]

        np.save(os.path.join(data_dir, 'train_'+str(f_id)+'.npy'), data)
        data = dict ()
        for keys, values in full_data.items():
            data[keys] = np.array( values)[vl_indices]
        np.save(os.path.join(data_dir, 'valid_'+str(f_id)+'.npy'), data)
        data = dict ()
        for keys, values in full_data.items():
            data[keys] = np.array( values)[ts_indices]
        np.save(os.path.join(data_dir, 'test_'+str(f_id)+'.npy'), data)
   
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument ('--domain', default='Positive', help='Enter a proper emotion')
    parser.add_argument ('--modal', default='all', help='Enter a proper emotion')
    parser.add_argument ('--speaker', default=1, help='Enter a proper emotion')
    args = parser.parse_args()
    speaker = int(args.speaker)
    print(args)
    
    if args.domain not in ['Positive','Aggressive', 'Dysphoric','strategy']:
        pdb.set_trace()
    else:
        save_dir= './result_' + args.domain + '_' + args.modal + '/'
        feature_dir = os.path.join(feature_dir, 'data_speaker_'+str(speaker) + '_'+args.modal)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(feature_dir).mkdir(parents=True, exist_ok=True)
        domain = args.domain

    
    #glove_matrix = np.load (os.path.join('/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/','glv_embedding_matrix_apo'),allow_pickle=True)
    #glove_matrix = np.load (os.path.join('/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan','glv_evmbedding_matrix'),allow_pickle=True)
    
    #embedding_path = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/'
    #embedding_path = '/etc/Sanchayan/'
    #glove_matrix = np.load (os.path.join(embedding_path,'glv_embedding_matrix_apo_full'),allow_pickle=True)
    #print (os.listdir(file_path))

    "Cuda related Operations"    
    cuda = torch.cuda.is_available() 
    #args.cuda = False
  
    if cuda:
        print('Running on GPU')
        device = torch.device ('cuda:2')
    else:
        print('Running on CPU')
        device = torch.device ('cpu')
    #device = torch.device('cpu')

    batch_size= 16
    n_classes = 2
    n_epochs=50


    
    #vocab_size = glove_matrix.shape[0]
    embedding_dim=300
   
  
    emo_dim=2 
    speaker_dim=2
    class_hid = 256

    class_weights =[0.7,0.3]
    
    '''
    #100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100

    D_a = 100 # concat attention
    '''

    config_list =['0.0001_0.0',
                    '0.0001_0.3',
                    '0.0001_0.5',
                    '0.0001_0.8']
    

    '''
    config_list = ['0.001_0.3', 
                   '0.0001_0.3',
                   '0.01_0.5',
                   '0.001_0.5',
                   '0.01_0.3', 
                   '0.0001_0.5']
    '''
    for f_id in range(10):

        #tr_indices , vl_indices , ts_indices = get_split_indices ( fold_dir = fold_dir, full_data = FULL_DATA, fold_no = f_id)

        train_loader, valid_loader, test_loader =\
                 get_TPOT_loaders(domain= domain, 
                    data = data_dir,
                    fold=f_id,
                    valid=0.1,
                    #indices = [tr_indices, vl_indices, ts_indices],
                    batch_size=batch_size,
                    num_workers=0,
                    speaker=speaker)

        best_all_config_loss= None 
        best_tr, best_vl, best_ts = None, None, None
        for config in config_list:
            lr_config, drop_config= config.split('_')
            

            #-----Data preparation -------#
                      
            #model = SiameseNet (MLP_Model( D_a+D_v , D_hid=class_hid, drop= float(drop_config)))
            model = MLP_general (D_a=31, D_v=17, D_hid=256, word_dict= wordmap,embed_size=64,drop =float(drop_config), modal=args.modal)

            model.to(device)
            
            
            
           
            
            loss_weights = torch.FloatTensor([class_weights]).to(device)
           
            #---------------Change the loss function to BCE Loss -------------------#
            #loss_function = MaskedNLLLoss(weight=loss_weights).to(device)
            loss_function = torch.nn.BCEWithLogitsLoss()
            optimizer= optim.Adam(model.parameters(), lr=float(lr_config), weight_decay = 0.01)
            
        
                  
            best_loss, best_label, best_pred, best_mask, best_probs = None, None, None, None, None
            best_epoch = None 

            train_losses=[]
            valid_losses=[]
            for e in range(n_epochs):
                start_time = time.time()
            
                
                train_loss= train_or_eval_model(model, loss_function,train_loader, e, optimizer=optimizer,device=device, train= True, speaker=speaker)
                valid_loss= train_or_eval_model(model, loss_function, valid_loader, e,device=device, speaker=speaker)
                fam,  preds, evals, strategy= test_model(model, loss_function, test_loader, e,device=device, speaker=speaker)
            
            

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)     
            
                if best_loss == None or best_loss > valid_loss:
                    '''
                    best_loss, best_label, best_pred, best_mask, best_attn =\
                            test_loss, test_label, test_pred, test_mask, attentions
                    '''
                    best_loss = valid_loss
                    best_epoch = e 
                 
                    #train_speaker_stack, train_listener_stack, train_name = get_embeddings (model, tr_em_loader,device=device)
                    #valid_speaker_stack, valid_listener_stack, valid_name = get_embeddings (model, vl_em_loader, device=device)
                    #test_speaker_stack, test_listener_stack, test_name = get_embeddings (model, ts_em_loader, device=device)

                    ts_res = make_data (speaker, preds, evals, strategy, fam)
                    
            
                
          
                
                print('epoch {} train_loss {}  valid_loss {}  time {}'.\
                        format(e+1, train_loss,  valid_loss,\
                                 round(time.time()-start_time,2)))
                if best_all_config_loss == None or best_all_config_loss > best_loss:
                    best_all_config_loss = best_loss
                  
                    best_ts = ts_res


    
         
            print (str(f_id) + '_'+ config, str(best_epoch))            

            # with open('best_attention.p','wb') as f:
            #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)
            
            #Write 
            #np.save (feature_dir + 'train_'+str(f_id)+'.npy', best_tr)
            #np.save (feature_dir + 'valid_'+str(f_id)+'.npy', best_vl)
            np.save (feature_dir + 'test_'+str(f_id)+'.npy', best_ts)
            #np.save (save_dir+str(f_id)+'_'+config+'xresult.npy', res)