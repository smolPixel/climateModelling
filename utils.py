import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
import copy
import re
#
class Envirodataset(Dataset):
	def __init__(self, data_file):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		file=open(data_file, 'r').readlines()
		for i, row in enumerate(file):
			print(row.keys())
			fds
			index+=1

	# def tokenize_and_vectorize(self, sentences):
	#     """Takes an array of sentences and return encoded data"""

	# def get_unlabelled(self):
	# 	dico={key:value for key, value in self.data.items() if value['label']==2}
	# 	return dico


	def tokenize(self, sentence):
		"Tokenize a sentence"
		return self.vocab_object(self.tokenizer.tokenize("<bos> " + sentence + " <eos>"))

	def reset_index(self):
		new_dat = {}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data = new_dat

	@property
	def vocab_size(self):
		return len(self.vocab_object)

	@property
	def vocab(self):
		return self.vocab_object.get_stoi()
		# fsd
		# return self.vocab_object

	def get_w2i(self):
		return self.vocab_object.get_stoi()

	def process_generated(self, exo):
		generated = idx2word(exo, i2w=self.get_i2w(),
							 pad_idx=self.get_w2i()['<pad>'],
							 eos_idx=self.get_w2i()['<eos>'])
		for sent in generated:
			print("------------------")
			print(sent)

	def get_i2w(self):
		return self.vocab_object.get_itos()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):

		input = self.data[item]['input'][:self.max_len]
		length= len(input)
		label = self.data[item]['label']
		input.extend([self.pad_idx] * (self.max_len - len(input)))
		target=input[1:]
		input=input[:-1]
		return {
			'sentence': self.data[item]['sentence'],
			'length': length,
			'input': np.asarray(input, dtype=int),
			'target': np.asarray(target, dtype=int),
			'label': label,
		}

	def get_texts(self):
		"""returns a list of the textual sentences in the dataset"""
		ll=[]
		for _, dat in self.data.items():
			ll.append(dat['sentence'])
		return ll

	def shape_for_loss_function(self, logp, target):
		target = target.contiguous().view(-1).cuda()
		logp = logp.view(-1, logp.size(2)).cuda()
		return logp, target

	def convert_tokens_to_string(self, tokens):
		if tokens==[]:
			return ""
		else:
			raise ValueError("Idk what this is supposed to return")

	def arr_to_sentences(self, array):
		if self.argdict['tokenizer'] in ['tweetTokenizer', 'PtweetTokenizer']:
			sentences=[]
			for arr in array:
				arr=arr.int()
				sent=self.vocab_object.lookup_tokens(arr.tolist())
				ss=""
				for token in sent:
					if token== "<bos>":
						continue
					if token =="<eos>":
						break
					ss+=f" {token}"
				sentences.append(ss)
			return sentences
		else:
			sent_token = self.tokenizer.batch_decode(array, skip_special_token=True)
			sent_str = []
			for i, ss in enumerate(sent_token):
				ss = ss.replace('<bos>', '')
				ss = ss.replace('<eos>', '')
				ss = ss.replace('[SEP].', '')
				ss = ss.replace('[SEP]', '')
				ss = re.sub(' +', ' ', ss)
				sent_token[i] = ss.strip()
			return sent_token
	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')