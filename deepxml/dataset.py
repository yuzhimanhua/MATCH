import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from typing import Sequence, Optional

TDataX = Sequence[Sequence]
TDataY = Optional[csr_matrix]

class MultiLabelDataset(Dataset):
	def __init__(self, data_x: TDataX, data_y: TDataY = None, training=True):
		self.data_x, self.data_y, self.training = data_x, data_y, training

	def __getitem__(self, item):
		data_x = self.data_x[item]
		if self.training and self.data_y is not None:
			data_y = self.data_y[item].toarray().squeeze(0).astype(np.float32)
			return data_x, data_y
		else:
			return data_x

	def __len__(self):
		return len(self.data_x)