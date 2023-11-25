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
	def __init__(self, data_file, argdict):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		file=open(data_file, 'r').readlines()
		for i, row in enumerate(file):
			row=json.loads(row)
			row={key: torch.tensor(it) for key, it in row.items()}
			self.data[len(self.data)]=row
			if argdict['short_data'] and len(self.data)>10:
				break

	def reset_index(self):
		new_dat = {}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data = new_dat

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		return self.data[item]

	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')