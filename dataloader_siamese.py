import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd

#from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pdb 

import numpy as np 
import itertools
import pymrmr
import random 
#from syntactic_features import *


opensmile_df =['GeMAPS_lld_Loudness_sma3', 'GeMAPS_lld_alphaRatio_sma3',
       'GeMAPS_lld_hammarbergIndex_sma3', 'GeMAPS_lld_slope0-500_sma3',
       'GeMAPS_lld_slope500-1500_sma3', 'GeMAPS_lld_spectralFlux_sma3',
       'GeMAPS_lld_mfcc1_sma3', 'GeMAPS_lld_mfcc2_sma3',
       'GeMAPS_lld_mfcc3_sma3', 'GeMAPS_lld_mfcc4_sma3',
       'GeMAPS_lld_F0semitoneFrom27.5Hz_sma3nz',
       'GeMAPS_lld_jitterLocal_sma3nz', 'GeMAPS_lld_shimmerLocaldB_sma3nz',
       'GeMAPS_lld_HNRdBACF_sma3nz', 'GeMAPS_lld_logRelF0-H1-H2_sma3nz',
       'GeMAPS_lld_logRelF0-H1-A3_sma3nz', 'GeMAPS_lld_F1frequency_sma3nz',
       'GeMAPS_lld_F1bandwidth_sma3nz',
       'GeMAPS_lld_F1amplitudeLogRelF0_sma3nz',
       'GeMAPS_lld_F2frequency_sma3nz',
       'GeMAPS_lld_F2amplitudeLogRelF0_sma3nz',
       'GeMAPS_lld_F3frequency_sma3nz',
       'GeMAPS_lld_F3amplitudeLogRelF0_sma3nz'
    ]
opensmile_csv_df =['GeMAPS_F0semitoneFrom27.5Hz_sma3nz_amean',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
       'GeMAPS_loudness_sma3_amean', 'GeMAPS_loudness_sma3_stddevNorm',
       'GeMAPS_loudness_sma3_percentile20.0',
       'GeMAPS_loudness_sma3_percentile50.0',
       'GeMAPS_loudness_sma3_percentile80.0',
       'GeMAPS_loudness_sma3_pctlrange0-2',
       'GeMAPS_loudness_sma3_meanRisingSlope',
       'GeMAPS_loudness_sma3_stddevRisingSlope',
       'GeMAPS_loudness_sma3_meanFallingSlope',
       'GeMAPS_loudness_sma3_stddevFallingSlope',
       'GeMAPS_spectralFlux_sma3_amean', 'GeMAPS_spectralFlux_sma3_stddevNorm',
       'GeMAPS_mfcc1_sma3_amean', 'GeMAPS_mfcc1_sma3_stddevNorm',
       'GeMAPS_mfcc2_sma3_amean', 'GeMAPS_mfcc2_sma3_stddevNorm',
       'GeMAPS_mfcc3_sma3_amean', 'GeMAPS_mfcc3_sma3_stddevNorm',
       'GeMAPS_mfcc4_sma3_amean', 'GeMAPS_mfcc4_sma3_stddevNorm',
       'GeMAPS_jitterLocal_sma3nz_amean',
       'GeMAPS_jitterLocal_sma3nz_stddevNorm',
       'GeMAPS_shimmerLocaldB_sma3nz_amean',
       'GeMAPS_shimmerLocaldB_sma3nz_stddevNorm',
       'GeMAPS_HNRdBACF_sma3nz_amean', 'GeMAPS_HNRdBACF_sma3nz_stddevNorm',
       'GeMAPS_logRelF0-H1-H2_sma3nz_amean',
       'GeMAPS_logRelF0-H1-H2_sma3nz_stddevNorm',
       'GeMAPS_logRelF0-H1-A3_sma3nz_amean',
       'GeMAPS_logRelF0-H1-A3_sma3nz_stddevNorm',
       'GeMAPS_F1frequency_sma3nz_amean',
       'GeMAPS_F1frequency_sma3nz_stddevNorm',
       'GeMAPS_F1bandwidth_sma3nz_amean',
       'GeMAPS_F1bandwidth_sma3nz_stddevNorm',
       'GeMAPS_F1amplitudeLogRelF0_sma3nz_amean',
       'GeMAPS_F1amplitudeLogRelF0_sma3nz_stddevNorm',
       'GeMAPS_F2frequency_sma3nz_amean',
       'GeMAPS_F2frequency_sma3nz_stddevNorm',
       'GeMAPS_F2bandwidth_sma3nz_amean',
       'GeMAPS_F2bandwidth_sma3nz_stddevNorm',
       'GeMAPS_F2amplitudeLogRelF0_sma3nz_amean',
       'GeMAPS_F2amplitudeLogRelF0_sma3nz_stddevNorm',
       'GeMAPS_F3frequency_sma3nz_amean',
       'GeMAPS_F3frequency_sma3nz_stddevNorm',
       'GeMAPS_F3bandwidth_sma3nz_amean',
       'GeMAPS_F3bandwidth_sma3nz_stddevNorm',
       'GeMAPS_F3amplitudeLogRelF0_sma3nz_amean',
       'GeMAPS_F3amplitudeLogRelF0_sma3nz_stddevNorm',
       'GeMAPS_alphaRatioV_sma3nz_amean',
       'GeMAPS_alphaRatioV_sma3nz_stddevNorm',
       'GeMAPS_hammarbergIndexV_sma3nz_amean',
       'GeMAPS_hammarbergIndexV_sma3nz_stddevNorm',
       'GeMAPS_slopeV0-500_sma3nz_amean',
       'GeMAPS_slopeV0-500_sma3nz_stddevNorm',
       'GeMAPS_slopeV500-1500_sma3nz_amean',
       'GeMAPS_slopeV500-1500_sma3nz_stddevNorm',
       'GeMAPS_spectralFluxV_sma3nz_amean',
       'GeMAPS_spectralFluxV_sma3nz_stddevNorm', 'GeMAPS_mfcc1V_sma3nz_amean',
       'GeMAPS_mfcc1V_sma3nz_stddevNorm', 'GeMAPS_mfcc2V_sma3nz_amean',
       'GeMAPS_mfcc2V_sma3nz_stddevNorm', 'GeMAPS_mfcc3V_sma3nz_amean',
       'GeMAPS_mfcc3V_sma3nz_stddevNorm', 'GeMAPS_mfcc4V_sma3nz_amean',
       'GeMAPS_mfcc4V_sma3nz_stddevNorm', 'GeMAPS_alphaRatioUV_sma3nz_amean',
       'GeMAPS_hammarbergIndexUV_sma3nz_amean',
       'GeMAPS_slopeUV0-500_sma3nz_amean',
       'GeMAPS_slopeUV500-1500_sma3nz_amean',
       'GeMAPS_spectralFluxUV_sma3nz_amean', 'GeMAPS_loudnessPeaksPerSec',
       'GeMAPS_VoicedSegmentsPerSec', 'GeMAPS_MeanVoicedSegmentLengthSec',
       'GeMAPS_StddevVoicedSegmentLengthSec',
       'GeMAPS_MeanUnvoicedSegmentLength',
       'GeMAPS_StddevUnvoicedSegmentLength',
       'GeMAPS_equivalentSoundLevel_dBp']


prosody_df =['prosodyAcf_voiceProb_sma', 'prosodyAcf_F0_sma',
       'prosodyAcf_pcm_loudness_sma']
vad_df =['vad_df']


covarep_df = ['covarep_vowelSpace',
            'covarep_MCEP_0',
            'covarep_MCEP_1',
            'covarep_VAD',
            'covarep_f0',
            'covarep_NAQ',
            'covarep_QOQ',
            'covarep_MDQ',
            'covarep_peakSlope',
            'covarep_F1',
            'covarep_F2']
covarep_names= ('Median of '+pd.Series(covarep_df) ).tolist() + ('IQR of' +pd.Series(covarep_df[4:])).tolist()
opensmile_names =('Median of '+pd.Series(opensmile_csv_df)).tolist() + ('Median of' + pd.Series(opensmile_df)). tolist() + ('IQR of'+ pd.Series(opensmile_df)).tolist() \
                + ('Median of '+ pd.Series(prosody_df)).tolist() + ('Median Of '+pd.Series(vad_df)).tolist() 


audio_names = covarep_names + opensmile_names

#-----------Definition of Video names -------------------------#
zface_feat = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','head_pitch','head_yaw','head_roll']
au_occ_feat = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
au_int_feat = ['int_6', 'int_10', 'int_12', 'int_14']

video_names= ('Max of ' + pd.Series(zface_feat)).tolist() \
            + ('Mean of ' + pd.Series(zface_feat)).tolist() \
            + ('Std of '+pd.Series(zface_feat)).tolist()\
            +  ('IQR of '+ pd.Series(zface_feat)).tolist() \
            + ('Median of ' + pd.Series(zface_feat)).tolist() \
            + ('Max of ' + pd.Series(au_occ_feat)).tolist() \
            + ('Mean of ' + pd.Series(au_occ_feat)).tolist()  \
            + ('Std of '+pd.Series(au_occ_feat)).tolist()  \
            + ('IQR of '+ pd.Series(au_occ_feat)).tolist() \
            + ('Median of ' + pd.Series(au_occ_feat)).tolist() \
            + ('Max of ' + pd.Series(au_int_feat)).tolist() \
            + ('Mean of' + pd.Series(au_int_feat)).tolist()  \
            + ('Std of'+pd.Series(au_int_feat)).tolist()  \
            + ('IQR of'+ pd.Series(au_int_feat)).tolist()\
            + ('Median of' + pd.Series(au_int_feat)).tolist()


#---Addition 
#video_names = video_names + ['IQR of gaze_x', 'IQR for gaze_y']
# Load the BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

    
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=128,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            truncation='longest_first',
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
    
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors


    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessor_idf (train, val, test):
        #without smooth IDF
    print("Without Smoothing:")
    #define tf-idf
    '''
     tf_idf_vec = TfidfVectorizer(use_idf=True, 
                            smooth_idf=False,  
                            ngram_range=(1,1),stop_words='english') # to use only  bigrams ngram_range=(2,2)
    #transform
    tf_idf_data = tf_idf_vec.fit_transform(text)
    
    #create dataframe
    tf_idf_dataframe=pd.DataFrame(tf_idf_data.toarray(),columns=tf_idf_vec.get_feature_names())
    print(tf_idf_dataframe)
    print("\n")
     
    #with smooth
    '''
    tf_idf_vec_smooth = TfidfVectorizer(use_idf=True,  
                            smooth_idf=True,  
                            ngram_range=(1,1),stop_words='english')
     
     
    tf_idf_data_smooth = tf_idf_vec_smooth.fit(train)

    train_dat = tf_idf_data_smooth.transform(train).toarray()
    val_dat = tf_idf_data_smooth.transform(val).toarray()
    test_dat = tf_idf_data_smooth.transform(test).toarray()

  
    pca = PCA(n_components = 200, svd_solver='full')
    pca_fit= pca.fit(train_dat)
    
    train_dat = pca_fit.transform(train_dat)
    val_dat = pca_fit.transform(val_dat)
    test_dat = pca_fit.transform(test_dat)
  
    return train_dat, val_dat, test_dat

def text_preprocessor (text):
    CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                       stop_words='english')
    count_data = CountVec.fit_transform(text)

    pdb.set_trace()
 

    return 
def analyze (feature, label, top_k=50):

    feature_set = label.reshape(-1,1)
    '''
    for idx,subset in enumerate(itertools.combinations (audio_names+video_names, top_k)):
      print (idx)
    '''
    
    if len (feature) > 1:
        for f in feature:
            feature_set = np.hstack ([feature_set, f]) if feature_set is not None else f 

    feature_name = ['label'] + audio_names + video_names
    
    df = pd.DataFrame( data = feature_set, index=None, columns= feature_name)

    x= pymrmr.mRMR(df, 'MIQ', top_k)
    pdb.set_trace()

    return 
def prune_features (audio, video, feature_name):

  #global_feature_names = audio_names + video_names
  
  a_n= np.array(audio_names)
  v_n = np.array(video_names)

  def read_feature_file(filename):
    df = pd.read_csv(filename, '\t', header=None, names=['Names', 'Score'])

    names = df['Names'].values.tolist()
    return names 

  names = read_feature_file(feature_name)
  names = names [:50]
  new_audio = []
  new_video = []
  
  for ele in names:

    if ele in a_n:
      try:  
        new_audio.append(audio [:, np.where(a_n == ele)[0][0]])
      except:
        pdb.set_trace()
    elif ele in v_n:
      try:
        new_video.append(video [:, np.where(v_n == ele)[0][0]])
      except:
        pdb.set_trace()
  new_audio= np.array(new_audio).T
  new_video= np.array(new_video).T

  return new_audio, new_video 


class TPOT_trainer(Dataset):

    def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=False, window_size=0.2, speaker=1):
        '''
        data  = dict()

        for keys, values in full_data.items():
          data[keys] = np.array( values)[select_indices]
        '''
        no_of_frames = int(60.0  / window_size)
        future_frames = int (3.0 /window_size)
        data = np.load(full_data, allow_pickle=True).item()
      
        sp2_audio = data['audio_mother']
        sp2_video = data['video_mother']

        sp1_audio = data['audio_child']
        sp1_video = data['video_child']

        audio_start = data['audio_start']
        audio_end = data['audio_end']
        
        sp1_token = data['token_child']
        sp2_token = data['token_mother']

        sp1_ling = data['ling_child']
        sp2_ling = data['ling_mother']

        sp1_props, sp2_props =[], []
        for val_1, val_2 in zip(sp1_ling, sp2_ling):
          sp1_props.append(val_1 ['props'])
          sp2_props.append(val_2 ['props'])
        sp1_props = np.array(sp1_props)
        sp2_props = np.array(sp2_props)
        if speaker == 1:
          sp1_labels = data['c_labels']
        else:
          sp1_labels = data['p_labels']
        family = data['family']
      
        
        #----------Now we extract the labels for next 3 seconds ------------#
        t_labels = []
        t1_audio, t1_video = [], []
        t2_audio, t2_video = [], []
        t_start, t_end =[], []
        t1_token,t2_token = [], []
        t1_props, t2_props = [],[]

        for idx, (end) in enumerate(audio_end):
          start = 0 
          label_vec = None
          while start + future_frames   < len(audio_end[idx]) :
            vec =  sp1_labels[idx] [start+1 : start+1 + future_frames]
            label_vec = np.vstack ([label_vec , vec]) if label_vec is not None else vec 
            
            start = start+ 1
          t_labels.append (label_vec)
          t1_audio.append (sp1_audio[idx][:len(label_vec)])
          t1_video.append (sp1_video[idx][:len(label_vec)])
          t2_audio.append (sp2_audio[idx][:len(label_vec)])
          t2_video.append (sp2_video[idx][:len(label_vec)])
          t_start.append (audio_start[idx][:len(label_vec)])
          t_end.append(audio_end[idx][:len(label_vec)])
          t1_props.append (sp1_props[idx][:len(label_vec)])
          t2_props.append (sp2_props[idx][:len(label_vec)])
          t1_token.append (sp1_token[idx][:len(label_vec)])
          t2_token.append (sp2_token[idx][:len(label_vec)])
      
        #-----------Now we create 1 minute insntances from these long videos --------- #
        self.family=[]
        self.labels= []
        self.sp1_audio, self.sp1_video = [], []
        self.sp2_audio, self.sp2_video = [], []
        self.start, self.end = [], []
        self.sp1_token, self.sp2_token= [],[]
        self.sp1_props, self.sp2_props= [],[]
        for idx , (end) in enumerate(audio_end):
          start =0 

          while start + no_of_frames < len(t_end[idx]):
            t_labels_ = t_labels[idx] [start: start + no_of_frames]
            t1_audio_ , t1_video_= t1_audio[idx] [start: start + no_of_frames], t1_video[idx][start:start+ no_of_frames]
            t2_audio_ , t2_video_= t2_audio[idx] [start: start + no_of_frames], t2_video[idx][start:start+ no_of_frames]
            t_start_, t_end_ = t_start[idx][start:start+no_of_frames] , t_end[idx][start:start+no_of_frames]
            t1_props_, t2_props_ = t1_props[idx][start:start+no_of_frames], t2_props[idx][start:start+no_of_frames]
            t1_token_, t2_token_ = t1_token[idx][start:start+no_of_frames] , t2_token[idx][start:start+no_of_frames]

            self.family.append(family[idx])
            self.labels.append(t_labels_)
            self.sp1_audio.append(t1_audio_)
            self.sp1_video.append(t1_video_)
            self.sp2_audio.append(t2_audio_)
            self.sp2_video.append(t2_video_)
            self.sp1_token.append(t1_token_)
            self.sp2_token.append(t2_token_)
            self.sp1_props.append(t1_props_)
            self.sp2_props.append(t2_props_)
            self.start.append(t_start_)
            self.end.append(t_end_)
            start += no_of_frames
        self.len = len(self.family)
        
        self.mask = self._set_ling ([self.sp1_token, self.sp2_token],\
                        [self.sp1_props, self.sp2_props], seq_len = len(self.sp1_audio[0]) )
        
        
        '''    
        self.len = len(self.labels)
        "Scaling the data"
        if train:
          self.scaler = self._audiovideo_(scaler=None)
        else:
          self._audiovideo_(scaler=scaler) 
        '''
    def _set_ling (self, tokens, props, seq_len):
      # len_mask = Contains the number of words per time frame
      sp1_token, sp2_token = tokens
      sp1_props, sp2_props = props
      max_len = len(sp1_token[0][0])
      for ii,num1 in enumerate(sp1_token):
        for jj,num2 in enumerate(num1):
          if max_len < len(num2):
            max_len = len(num2)
      len_mask = np.zeros ((self.len, 300))
      sp1_token_, sp2_token_ = [], []
      sp1_props_, sp2_props_ = [], []

      for ii in range (300):
        sp1_token_stack =[]
        sp1_props_stack =[]
        sp2_token_stack =[]
        sp2_props_stack =[]
        for jj, num2 in enumerate (sp1_token):
          
          sp1_token_stack.append(sp1_token [jj][ii].tolist() + [0]* (max_len - len(sp1_token[jj][ii])))
          sp2_token_stack.append(sp2_token [jj][ii].tolist() + [0]* (max_len - len(sp2_token[jj][ii])))
          sp1_props_stack.append(sp1_props [jj][ii].tolist() + [0]* (max_len - len(sp1_props[jj][ii])))
          sp2_props_stack.append(sp2_props [jj][ii].tolist() + [0]* (max_len - len(sp2_props[jj][ii])))
          len_mask [jj][ii] = len(sp1_token[jj][ii])
        sp1_token_.append(sp1_token_stack)
        sp2_token_.append(sp2_token_stack)
        sp1_props_.append(sp1_props_stack)
        sp2_props_.append(sp2_props_stack)
      
      self.sp1_token = np.stack(sp1_token_).transpose(1,0,2)
      self.sp2_token = np.stack(sp2_token_).transpose(1,0,2)
      self.sp1_props = np.stack(sp1_props_).transpose(1,0,2)
      self.sp2_props = np.stack(sp2_props_).transpose(1,0,2)


      return len_mask
      
    def _audiovideo_(self, scaler=None):
      if scaler is None :
        audio_scaler = StandardScaler()
        video_scaler = StandardScaler()
        l_audio_scaler= StandardScaler()
        l_video_scaler= StandardScaler()
        

        fit_audio = audio_scaler.fit(self.audio)
        fit_video = video_scaler.fit(self.video)
        fit_l_audio = l_audio_scaler.fit(self.l_audio)
        fit_l_video = l_video_scaler.fit(self.l_video)

        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
        self.l_audio =fit_l_audio.transform(self.l_audio)
        self.l_video =fit_l_video.transform(self.l_video)

        return [fit_audio, fit_video, fit_l_audio, fit_l_video]

      else:
        [fit_audio,fit_video, fit_l_audio, fit_l_video] = scaler
        
        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
        self.l_audio =fit_l_audio.transform(self.l_audio)
        self.l_video =fit_l_video.transform(self.l_video)

      
      
    def __getitem__(self, index):
    

        return torch.FloatTensor (self.sp1_audio[index]),\
               torch.FloatTensor (self.sp1_video[index]),\
               torch.FloatTensor (self.sp2_audio[index]),\
               torch.FloatTensor (self.sp2_video[index]),\
               torch.LongTensor (self.sp1_token[index]), torch.LongTensor(self.sp2_token[index]), \
               torch.FloatTensor(self.sp1_props[index]), torch.FloatTensor(self.sp2_props[index]),\
               torch.FloatTensor(self.labels[index]),\
               torch.LongTensor(self.mask[index]),\
               self.family[index]
          

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return pad_sequence(dat[0],True), pad_sequence(dat[1],True), pad_sequence (dat[2],True), pad_sequence(dat[3],True),\
            pad_sequence(dat[8],True),\
            pad_sequence(dat[4],True),pad_sequence(dat[5],True), pad_sequence(dat[6],True), pad_sequence(dat[7],True),  pad_sequence(dat[9],True),\
            dat[9].tolist()


class TPOT_tester(Dataset):

    def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=False, window_size=0.2, speaker=1):
        
        '''
        data  = dict()

        for keys, values in full_data.items():
          data[keys] = np.array( values)[select_indices]
        '''
        no_of_frames = int(60.0  / window_size)
        future_frames = int (3.0 /window_size)
        data = np.load(full_data[0], allow_pickle=True).item()
        data2= np.load(full_data[1], allow_pickle=True).item()
        sp2_audio = np.concatenate([data['audio_mother'], data2['audio_mother']])
        sp2_video = np.concatenate([data['video_mother'], data2['video_mother']])
        
        sp1_audio = np.concatenate([data['audio_child'], data2['audio_child']]) 
        sp1_video = np.concatenate([data['video_child'], data2['video_child']])

        audio_start = np.concatenate([data['audio_start'], data2['audio_start']])
        audio_end = np.concatenate([data['audio_end'], data2['audio_end']])
        
        eval_labels = np.concatenate([data['eval_labels'], data2['eval_labels']])
        strategy_labels = np.concatenate([data['strategy_labels'], data2['strategy_labels']])

        sp1_token = np.concatenate([data['token_child'] ,data2['token_child']])
        sp2_token = np.concatenate([data['token_mother'] ,data2['token_mother']])
    
        sp1_ling = np.concatenate([data['ling_child'], data2['ling_child']])
        sp2_ling = np.concatenate([data['ling_mother'], data2['ling_mother']])
    
        sp1_props, sp2_props =[], []
        for val_1, val_2 in zip(sp1_ling, sp2_ling):
          sp1_props.append(val_1 ['props'])
          sp2_props.append(val_2 ['props'])
        sp1_props = np.array(sp1_props)
        sp2_props = np.array(sp2_props)
        
        if speaker == 1:

          sp1_labels = np.concatenate([data['c_labels'], data2['c_labels']])
        else:
          sp1_labels = np.concatenate([data['p_labels'], data2['p_labels']])

        family = np.concatenate([data['family'],data2['family']])
        
        #-----------Now we create evaluation datapoints --------- #
        self.family=[]
        self.labels, self.strategy= [],[]
        self.sp1_audio, self.sp1_video = [], []
        self.sp2_audio, self.sp2_video = [], []
        self.start, self.end = [], []
        self.sp1_token, self.sp2_token= [],[]
        self.sp1_props, self.sp2_props= [],[]

        for idx , (end) in enumerate(audio_end):

          

          indices = np.where (eval_labels[idx] != 0)[0]

          for j_idx, index in enumerate(indices):
            self.sp1_audio.append(sp1_audio [idx] [index-5: index+1])
            self.sp2_audio.append (sp2_audio [idx] [index-5: index+1])
            self.sp1_video.append (sp1_video[idx] [index-5: index+1])
            self.sp2_video.append (sp2_video[idx][index-5:index+1])
            self.sp1_token.append (sp1_token[idx][index-5:index+1])
            self.sp2_token.append (sp2_token[idx][index-5:index+1])
            self.sp1_props.append (sp1_props[idx][index-5:index+1])
            self.sp2_props.append (sp2_props[idx][index-5:index+1])
            self.start.append (audio_start[idx][index-5:index+1])
            self.end.append(audio_end[idx][index-5:index+1])
            self.labels.append([eval_labels[idx][index]])
            self.strategy.append([strategy_labels[idx][index]])
            self.family.append(family[idx])
        self.sp1_audio, self.sp2_audio = np.stack(self.sp1_audio), np.stack(self.sp2_audio)
        self.sp1_video, self.sp2_video = np.stack(self.sp2_video), np.stack(self.sp2_video)
        #self.sp1_token, self.sp2_token = np.stack(self.sp1_token), np.stack(self.sp2_token)
        #self.sp1_props, self.sp2_props = np.stack(self.sp1_props), np.stack(self.sp2_props)
        self.start, self.end = np.stack(self.start), np.stack(self.end)
        self.labels, self.strategy = np.concatenate(self.labels), np.concatenate(self.strategy)
        self.family = np.array(self.family)
      
        self.len = len(self.family)
        
        self.mask = self._set_ling ([self.sp1_token, self.sp2_token],\
                        [self.sp1_props, self.sp2_props], seq_len = 6 )
        
      
        '''    
        self.len = len(self.labels)
        "Scaling the data"
        if train:
          self.scaler = self._audiovideo_(scaler=None)
        else:
          self._audiovideo_(scaler=scaler) 
        '''
    def _set_ling (self, tokens, props, seq_len):
      # len_mask = Contains the number of words per time frame
      sp1_token, sp2_token = tokens
      sp1_props, sp2_props = props
      max_len = len(sp1_token[0][0])
      for ii,num1 in enumerate(sp1_token):
        for jj,num2 in enumerate(num1):
          if max_len < len(num2):
            max_len = len(num2)
      len_mask = np.zeros ((self.len, seq_len))
      sp1_token_, sp2_token_ = [], []
      sp1_props_, sp2_props_ = [], []

      for ii in range (seq_len):
        sp1_token_stack =[]
        sp1_props_stack =[]
        sp2_token_stack =[]
        sp2_props_stack =[]
        for jj, num2 in enumerate (sp1_token):
          
          sp1_token_stack.append(sp1_token [jj][ii].tolist() + [0]* (max_len - len(sp1_token[jj][ii])))
          sp2_token_stack.append(sp2_token [jj][ii].tolist() + [0]* (max_len - len(sp2_token[jj][ii])))
          sp1_props_stack.append(sp1_props [jj][ii].tolist() + [0]* (max_len - len(sp1_props[jj][ii])))
          sp2_props_stack.append(sp2_props [jj][ii].tolist() + [0]* (max_len - len(sp2_props[jj][ii])))
          len_mask [jj][ii] = len(sp1_token[jj][ii])
        sp1_token_.append(sp1_token_stack)
        sp2_token_.append(sp2_token_stack)
        sp1_props_.append(sp1_props_stack)
        sp2_props_.append(sp2_props_stack)
      
      self.sp1_token = np.stack(sp1_token_).transpose(1,0,2)
      self.sp2_token = np.stack(sp2_token_).transpose(1,0,2)
      self.sp1_props = np.stack(sp1_props_).transpose(1,0,2)
      self.sp2_props = np.stack(sp2_props_).transpose(1,0,2)


      return len_mask
      
    def _audiovideo_(self, scaler=None):
      if scaler is None :
        audio_scaler = StandardScaler()
        video_scaler = StandardScaler()
        l_audio_scaler= StandardScaler()
        l_video_scaler= StandardScaler()
        

        fit_audio = audio_scaler.fit(self.audio)
        fit_video = video_scaler.fit(self.video)
        fit_l_audio = l_audio_scaler.fit(self.l_audio)
        fit_l_video = l_video_scaler.fit(self.l_video)

        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
        self.l_audio =fit_l_audio.transform(self.l_audio)
        self.l_video =fit_l_video.transform(self.l_video)

        return [fit_audio, fit_video, fit_l_audio, fit_l_video]

      else:
        [fit_audio,fit_video, fit_l_audio, fit_l_video] = scaler
        
        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
        self.l_audio =fit_l_audio.transform(self.l_audio)
        self.l_video =fit_l_video.transform(self.l_video)

      
      
    def __getitem__(self, index):
    

        return torch.FloatTensor (self.sp1_audio[index]),\
               torch.FloatTensor (self.sp1_video[index]),\
               torch.FloatTensor (self.sp2_audio[index]),\
               torch.FloatTensor (self.sp2_video[index]),\
               torch.LongTensor (self.sp1_token[index]), torch.LongTensor(self.sp2_token[index]), \
               torch.FloatTensor(self.sp1_props[index]), torch.FloatTensor(self.sp2_props[index]),\
               self.labels[index],\
               self.strategy[index],\
               torch.LongTensor(self.mask[index]),\
               self.family[index]
          

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)


        
        return pad_sequence(dat[0],True), pad_sequence(dat[1],True), pad_sequence (dat[2],True), pad_sequence(dat[3],True),\
            torch.LongTensor(dat[8]), torch.LongTensor(dat[9]),\
            pad_sequence(dat[4],True),pad_sequence(dat[5],True), pad_sequence(dat[6],True), pad_sequence(dat[7],True),  pad_sequence(dat[10],True),\
            dat[11].tolist()

class TPOTDataset(Dataset):

    def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=False):

        data = np.load(full_data, allow_pickle=True).item()

        #self.audio_features= np.load(audio_path, allow_pickle=True).item()
        #self.video_features= np.load(video_path, allow_pickle=True).item()
        #self.text_features = np.load (text_path, allow_pickle=True).item()

        audio = data['audio']
        video = data['video']
        l_audio= data['listener_audio']
        l_video= data['listener_video']
        text = data['turn_text']
        text_seq  = data['turn_sequence']
        #label = data['turn_label'] 
        strategy = data['turn_strategy']
        speaker = data['turn_speaker']
        family = data['turn_filename']
        duration = data['turn_duration']   
        #gap = data['speaker_gap']  
          
        #self.tokenizer = tokenizer
        #self.max_len= 160        
        
        #----------Now we have to unpack from dialog level to turn level ------------#
        self.audio= np.concatenate(audio)
        self.video= np.concatenate(video)
        self.text = np.concatenate(text)
        self.text_seq = np.concatenate(text_seq)
        #self.label = np.concatenate(label)
        self.strategy = np.concatenate(strategy)
        self.speaker = np.concatenate(speaker)
        self.family = np.concatenate([ [family[i]] * len(x) for i,x in enumerate(audio) ])
        self.duration= np.concatenate(duration)
        self.l_video = np.concatenate(l_video)
        self.l_audio = np.concatenate(l_audio)
 
   

        self.audio, self.video = prune_features(self.audio, self.video, 'feat.txt')
        self.l_audio, self.l_video = prune_features (self.l_audio, self.l_video, 'feat.txt') 
   
        self.len = len(self.family)
        "Scaling the data"
        if train:
          self.scaler = self._audiovideo_(scaler=None)
        else:
          self._audiovideo_(scaler=scaler) 


      
    def _audiovideo_(self, scaler=None):
      if scaler is None :
        audio_scaler = StandardScaler()
        video_scaler = StandardScaler()
        

        fit_audio = audio_scaler.fit(self.audio)
        fit_video = video_scaler.fit(self.video)
        
        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
        return [fit_audio, fit_video]

      else:
        [fit_audio,fit_video] = scaler
        
        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
 
    def __getitem__(self, index):
        
        return torch.FloatTensor (self.audio[index]),\
               torch.FloatTensor (self.video[index]),\
               torch.FloatTensor (self.l_audio[index]), \
               torch.FloatTensor (self.l_video[index]), \
               self.strategy[index], \
               self.family[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        # Lengths of the individual sequences ----#
 
        #text = [ torch.LongTensor(x) for x in dat[3]]
        label = torch.LongTensor ([d for d in dat[4]]) #4 is srategy

      
        
        return pad_sequence(dat[0],True), pad_sequence(dat[1],True),\
             pad_sequence (dat[2],True), pad_sequence(dat[3],True), label, dat[5].tolist()

class TPOTDataset_MLP(Dataset):

    def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=False):
        [data_dir, feat_dir] = full_data
        data = np.load(data_dir, allow_pickle=True).item()

        feat = np.load (feat_dir, allow_pickle=True).item()

        #self.audio_features= np.load(audio_path, allow_pickle=True).item()
        #self.video_features= np.load(video_path, allow_pickle=True).item()
        #self.text_features = np.load (text_path, allow_pickle=True).item()
      
        speaker_track = feat['speaker']
        listener_track = feat['listener']
        
        text = data['turn_text']
        text_seq  = data['turn_sequence']
        #label = data['turn_label'] 
        strategy = data['turn_strategy']
        speaker = data['turn_speaker']
        family = data['turn_filename']
        duration = data['turn_duration']   
        #gap = data['speaker_gap']  
          
        #self.tokenizer = tokenizer
        #self.max_len= 160        
        
        #----------Now we have to unpack from dialog level to turn level ------------#
        self.feat = np.concatenate(speaker_track)
        self.l_feat = np.concatenate(listener_track)
      
        self.text = np.concatenate(text)
        self.text_seq = np.concatenate(text_seq)
        #self.label = np.concatenate(label)
        self.strategy = np.concatenate(strategy)
        self.speaker = np.concatenate(speaker)
        self.family = np.concatenate([ [family[i]] * len(x) for i,x in enumerate(speaker_track) ])
        self.duration= np.concatenate(duration)


        #---------------Selection 
        indices = self.strategy != 2
        self.feat= self.feat[indices]
        self.l_feat = self.l_feat [indices]
        self.text = self.text [indices]
        self.text_seq= self.text_seq [indices]
        self.strategy = self.strategy[indices]
        self.family = self.family[indices]
        self.duration = self.duration[indices]

      
        text_df = pd.DataFrame (self.text, columns =['feature'])
        text_feat = lexical_features (text_df, 'feature', self.duration)
        self.text = text_feat
    
      
        #self.audio, self.video = prune_features(self.audio, self.video, 'feat.txt')
        #self.l_audio, self.l_video = prune_features (self.l_audio, self.l_video, 'feat.txt') 
   
        self.len = len(self.family)
        "Scaling the data"
        if train:
          self.scaler = self._audiovideo_(scaler=None)
        else:
          self._audiovideo_(scaler=scaler) 


      
    def _audiovideo_(self, scaler=None):
      if scaler is None :
        text_scaler = StandardScaler()
      
        fit_text = text_scaler.fit(self.text)
        
        self.text= fit_text.transform(self.text)
      
        return [fit_text]

      else:
        [fit_text] = scaler
        
        self.text= fit_text.transform(self.text)
 
    def __getitem__(self, index):
        
        return torch.FloatTensor (self.feat[index]),\
               torch.FloatTensor (self.l_feat[index]),\
               torch.FloatTensor (self.text[index]),\
               self.strategy[index], \
               self.family[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        # Lengths of the individual sequences ----#
 
        #text = [ torch.LongTensor(x) for x in dat[3]]
        label = torch.LongTensor ([d for d in dat[3]]) #4 is srategy

      
      
        return pad_sequence(dat[0],True), pad_sequence(dat[1],True),\
             pad_sequence (dat[2],True), label, dat[4].tolist()
    


