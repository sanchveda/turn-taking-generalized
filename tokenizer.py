import os 
import numpy as np 
import json
import pdb 

filename = 'general_data.npy'
data = np.load (filename, allow_pickle=True).item()


ling_child = data['ling_child']
ling_mother = data['ling_mother']


child_words = []
mother_words =[]
for val_c, val_p in zip (ling_child,ling_mother):
	c_text = val_c['text']
	p_text = val_p['text']
	child_words.extend (c_text)
	mother_words.extend(p_text)
total_words = np.concatenate(child_words + mother_words)

unique_words = np.unique(total_words)

word_map = {k: v for v, k in enumerate(unique_words)}

with open('wordmap.json', 'w') as f:
	json.dump(word_map, f)


def map_words (text):
	return np.array([word_map[k] for k  in text])

token_child = []
token_mother = [] 	 
for val_c, val_p in zip(ling_child,  ling_mother):

	c_text= val_c ['text']
	p_text= val_p ['text']
	c_token = [map_words (w) for w in  c_text]
	p_token = [map_words (w) for w in  p_text]
	token_child.append (c_token)
	token_mother.append (p_token)

data['token_child']  = np.array(token_child)
data['token_mother'] = np.array(token_mother)

print ("Done")