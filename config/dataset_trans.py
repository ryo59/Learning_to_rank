# -- coding: utf-8 --

import os
import numpy  as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import torch
import torch.utils.data as data

SCALER_LEVEL = ['QUERY', 'DATASET']
SCALER_ID    = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']

MSLETOR       = ['MQ2007_Super', 'MQ2008_Super', 'MQ2007_Semi', 'MQ2008_Semi', 'MQ2007_List', 'MQ2008_List']
MSLETOR_SUPER = ['MQ2007_Super', 'MQ2008_Super']
MSLETOR_SEMI  = ['MQ2007_Semi', 'MQ2008_Semi']
MSLETOR_LIST  = ['MQ2007_List', 'MQ2008_List']
MSLRWEB       = ['MSLRWEB10K', 'MSLRWEB30K']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pickle

## due to the restriction of 4GB ##
max_bytes = 2**31 - 1

def pickle_save(target, file):
	bytes_out = pickle.dumps(target, protocol=4)
	with open(file, 'wb') as f_out:
		for idx in range(0, len(bytes_out), max_bytes):
			f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load(file):
	file_size = os.path.getsize(file)
	with open(file, 'rb') as f_in:
		bytes_in = bytearray(0)
		for _ in range(0, file_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	data = pickle.loads(bytes_in)
	return data

class L2RDataLoader():
	"""
	An abstract loader for learning-to-rank datasets
	"""
	def __init__(self, file, buffer=True):
		'''
		:param file:   the specified data file, e.g., the fold path when performing k-fold cross validation
		:param buffer: buffer the primarily parsed data
		'''
		self.df = None
		self.file = file
		self.buffer = buffer

	def load_data(self):
		pass

	def filter(self):
		pass



class MSL2RDataLoader(L2RDataLoader):
	"""
	The data loader for MS learning-to-rank datasets
	"""

	def __init__(self, file, data_id=None, buffer=True):
		super(MSL2RDataLoader, self).__init__(file=file, buffer=buffer)

		self.data_id = data_id
		# origianl data as dataframe
		self.df_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '.df' # the original data file buffer as a dataframe

		pq_suffix = 'PerQ'

		# plus scaling
		self.scale_data   = True
		self.scaler_id    = 'StandardScaler'

		if self.scale_data:
			pq_suffix = '_'.join([pq_suffix, 'QS', self.scaler_id])

		self.perquery_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' + pq_suffix +'.np'



	def load_data(self):
		'''
		Load data at a per-query unit consisting of {scaled} {des-sorted} document vectors and standard labels
		:param given_scaler: scaler learned over entire training data, which is only needed for dataset-level scaling
		:return:
		'''
		if self.data_id in MSLETOR:
			self.num_features = 46
		elif self.data_id in MSLRWEB:
			self.num_features = 136

		self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]

		if os.path.exists(self.perquery_file):
			list_Qs = pickle_load(self.perquery_file)
			return list_Qs
		else:
			self.get_df_file()

			self.ini_scaler()

			list_Qs = []
			qids = self.df.qid.unique()
			np.random.shuffle(qids)
			for qid in qids:
				sorted_qdf = self.df[self.df.qid == qid].sort_values('rele_truth', ascending=False)
				# if sorted_qdf["rele_truth"].isin([1.0]).any():
				doc_reprs  = sorted_qdf[self.feature_cols].values
				if self.scale_data:
					doc_reprs = self.scaler.fit_transform(doc_reprs)

				doc_labels = sorted_qdf['rele_truth'].values

				#doc_ids    = sorted_qdf['#docid'].values # commented due to rare usage

				list_Qs.append((qid, doc_reprs, doc_labels))
				# else:
				# 	pass

			if self.buffer: pickle_save(list_Qs, file=self.perquery_file)

			return list_Qs

	def get_df_file(self):
		''' Load original data file as a dataframe. If buffer exists, load buffered file. '''

		if os.path.exists(self.df_file):
			self.df = pd.read_pickle(self.df_file)
		else:
			if self.data_id in MSLETOR:
				self.df = self.load_LETOR4()
			elif self.data_id in MSLRWEB:
				self.df = self.load_MSLRWEB()

			if self.buffer:
				parent_dir = Path(self.df_file).parent
				if not os.path.exists(parent_dir): os.makedirs(parent_dir)
				self.df.to_pickle(self.df_file)



	def load_LETOR4(self):
		'''  '''
		df = pd.read_csv(self.file, sep=" ", header=None)
		df.drop(columns=df.columns[[-2, -3, -5, -6, -8, -9]], axis=1, inplace=True)  # remove redundant keys
		#print(self.num_features, len(df.columns) - 5)
		assert self.num_features == len(df.columns) - 5



		for c in range(1, self.num_features+2):           							 # remove keys per column from key:value
			df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

		df.columns = ['rele_truth', 'qid'] + self.feature_cols + ['#docid', 'inc', 'prob']

		if self.data_id in MSLETOR_SEMI and self.data_dict['unknown_as_zero']:
			self.df[self.df[self.feature_cols]<0] = 0

		for c in ['rele_truth'] + self.feature_cols:
			df[c] = df[c].astype(np.float32)

		df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering

		return df


	def load_MSLRWEB(self):
		'''  '''
		df = pd.read_csv(self.file, sep=" ", header=None)
		df.drop(columns=df.columns[-1], inplace=True) # remove the line-break
		assert self.num_features == len(df.columns) - 2

		for c in range(1, len(df.columns)):           # remove the keys per column from key:value
			df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])


		df.columns = ['rele_truth', 'qid'] + self.feature_cols

		for c in ['rele_truth'] + self.feature_cols:
			df[c] = df[c].astype(np.float32)

		df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)     # additional binarized column for later filtering

		return df


	def ini_scaler(self):
		assert self.scaler_id in SCALER_ID
		if self.scaler_id == 'MinMaxScaler':
			self.scaler = MinMaxScaler()
		elif self.scaler_id == 'RobustScaler':
			self.scaler = RobustScaler()
		elif self.scaler_id == 'StandardScaler':
			self.scaler = StandardScaler()





class L2RDataset(data.Dataset):
	'''
	Buffering tensored objects can save much time.
	'''
	def __init__(self, file, data_id):
		loader = MSL2RDataLoader(file=file, data_id=data_id)
		perquery_file = loader.perquery_file

		torch_perquery_file = perquery_file.replace('.np', '.torch')

		if os.path.exists(torch_perquery_file):
			self.list_torch_Qs = pickle_load(torch_perquery_file)
		else:
			self.list_torch_Qs = []

			list_Qs = loader.load_data()
			list_inds = list(range(len(list_Qs)))
			for ind in list_inds:
				qid, doc_reprs, doc_labels = list_Qs[ind]

				torch_batch_rankings = torch.from_numpy(doc_reprs).type(torch.FloatTensor)

				torch_batch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)

				self.list_torch_Qs.append((qid, torch_batch_rankings, torch_batch_std_labels))

			#buffer
			pickle_save(self.list_torch_Qs, torch_perquery_file)


	def __getitem__(self, index):
		qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
		return torch_batch_rankings, torch_batch_std_labels

	def __len__(self):
		return len(self.list_torch_Qs)
