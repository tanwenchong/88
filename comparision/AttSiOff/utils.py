from scipy import stats
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score
import sklearn
import random
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch.backends.cudnn as cudnn
import pandas as pd

def log(filename='logging.txt'):
	pcc, spcc, mcc, auc, prc = [], [], [], [], []

	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			line = line.split(',')
			pcc.append(float(line[0].split(':')[-1].strip()))
			spcc.append(float(line[1].split(':')[-1].strip()))
			# mcc.append(float(line[2].split(':')[-1].strip()))
			auc.append(float(line[2].split(':')[-1].strip()))
			# prc.append(float(line[4].split(':')[-1].strip()))
				

	data = pd.DataFrame()
	data['pcc'], data['spcc'], data['auc'] = pcc, spcc, auc
	data.to_csv(filename+'result.csv', index=0)

def init_seed(SEED):
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	cudnn.enabled = True
	cudnn.deterministic = True
	cudnn.benchmark = False

def compute_roc_curve(truth, prediction, thred = 0.7, if_save_img=False, title=None, save_path=None):
	truth = (truth>thred)
	# 标签必须是二进制

	if (truth.sum()==0 or truth.sum()==truth.shape[0]):
		return -1

	result = roc_auc_score(truth, prediction)

	if if_save_img:
		ticks_top = np.linspace(0,1,51) #0-1范围上，总共有51条线，即划分了50小份
		fpr, tpr, thresholds = roc_curve(truth, prediction, pos_label=1)
		# figure是plt中最基础的对象，是一个总的画布
		# add_subplot是在画布中添加一个子区域并返回
		# ax.plot 就是在该子区域内画图
		# plt.plot实际上就是获取当前的ax然后作图
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(fpr, tpr, linewidth=2)
		ax.set_xticks(ticks_top, minor=True)
		ax.set_yticks(ticks_top, minor=True)
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title(title+str(round(result, 3)))
		ax.grid(which="minor",alpha=0.5) # 次要网格线的颜色深度0.5
		ax.grid(which="major", alpha=0.7) # 主要网格线深度0.7，更深
		ax.set_xlim(xmin=0, xmax=1)
		ax.set_ylim(ymin=0, ymax=1)
		plt.savefig(os.path.join(save_path, str(title)+str(round(result, 3))+'.png'))
		plt.close()

	return result

def compute_prc_curve(truth, prediction, thred = 0.7, if_save_img=False, title=None, save_path=None):
	truth = (truth>thred)
	# 标签必须是二进制
	# 精准召回曲线下面积AUPRC更适合评估不平衡数据集

	if (truth.sum()==0 or truth.sum()==truth.shape[0]):
		return -1

	result = sklearn.metrics.average_precision_score(truth, prediction)

	if if_save_img:
		ticks_top = np.linspace(0,1,51) #0-1范围上，总共有51条线，即划分了50小份
		precision, recall, thresholds = sklearn.metrics.precision_recall_curve(truth, prediction)
		# figure是plt中最基础的对象，是一个总的画布
		# add_subplot是在画布中添加一个子区域并返回
		# ax.plot 就是在该子区域内画图
		# plt.plot实际上就是获取当前的ax然后作图
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(recall, precision, linewidth=2)
		ax.set_xticks(ticks_top, minor=True)
		ax.set_yticks(ticks_top, minor=True)
		ax.set_xlabel('recall')
		ax.set_ylabel('precision')
		ax.set_title(title+str(round(result, 3)))
		ax.grid(which="minor",alpha=0.5) # 次要网格线的颜色深度0.5
		ax.grid(which="major", alpha=0.7) # 主要网格线深度0.7，更深
		ax.set_xlim(xmin=0, xmax=1)
		ax.set_ylim(ymin=0, ymax=1)
		plt.savefig(os.path.join(save_path, str(title)+str(round(result, 3))+'.png'))
		plt.close()

	return result


def compute_pcc_spcc(x,y):
	return round(stats.pearsonr(x,y)[0], 3), round(stats.spearmanr(x,y)[0], 3)

def compute_mcc(x,y):
	# 标签和预测均为二进制
	x=(x>0.7)
	y=(y>0.7)
	return matthews_corrcoef(x,y)


def plot_scatter_curve(pred, label, save_path, title):

	plt.scatter(pred, label, s=10, c='b') #点的大小默认20
	plt.title('exp vs pred / {}'.format(title))
	plt.xlabel('prediction')
	plt.ylabel('experiment')
	plt.xlim(0., 1.)
	plt.ylim(0., 1.)
	plt.savefig(os.path.join(save_path, str(title) + '.png'))
	plt.close()

def plot_metric_curve(metrics_dict, train_or_valid, base_path, epoches: tuple):
	x_axis = np.arange(epoches[0], epoches[1])

	for key in metrics_dict.keys():
		value = metrics_dict[key]
		plt.plot(x_axis, value, c='r')
		plt.title(train_or_valid + '_' + key)
		plt.xlabel('epoches')
		plt.ylabel(key)
		plt.savefig(os.path.join(base_path, train_or_valid + '_' + key + '.png'))
		plt.close()




def readFa(fa):
	'''
	@param fa {str}	 fasta 文件路径
	@return: {generator} 返回一个生成器，能迭代得到fasta文件的每一个序列名和序列
	注意，fna文件中有XM开头也有NM开头的，XM不太好，是计算机模拟生生的，而NM是有实验数据支撑的
	'''
	with open(fa,'r') as FA:
		seqName,seq='',''
		while 1:
			line=FA.readline()
			line=line.strip('\n')
			if (line.startswith('>') or not line) and seqName:
				yield ((seqName, seq))
			if line.startswith('>'):
				line=line.split(' ')
				seqName = line[1].strip()
				seq = ''
			else:
				seq += line
			if not line:break


def readFaRNAFOLD(fa):
	'''
	加载预处理好的mrna全长的rnafold预测结果
	'''
	with open(fa,'r') as FA:
		seqName,seq='',''
		while 1:
			line=FA.readline()
			line=line.strip('\n')
			if (line.startswith('>') or not line) and seqName:
				yield ((seqName, seq, min_energy))
			if line.startswith('>'):
				line=line.split(' ')
				seqName, min_energy = line[1].strip(), line[2].strip()
				seq = ''
			else:
				seq += line
			if not line:break


class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, model_save_path, model_save_name, patience=15, greater=False, delta=0):
		"""
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			greater (bool): the greater, the better.
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
		"""
		self.patience = patience
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.delta = delta
		self.greater = greater
		self.path = model_save_path
		self.best_model_path = None
		self.model_name = model_save_name

	def __call__(self, score, iter, model):

		if self.greater:
			cur_score = score
		else:
			cur_score = -score

		if self.best_score is None:
			self.best_score = cur_score
			self.save_checkpoint(score, model, iter)
		elif cur_score < self.best_score + self.delta:
			self.counter += 1
			if self.counter >= self.patience:
				# logging(f'EarlyStopping counter: {self.counter} out of {self.patience}')
				self.early_stop = True
		else:
			self.best_score = cur_score
			self.save_checkpoint(score, model, iter)
			self.counter = 0

	def save_checkpoint(self, score, model, iter):
		# self.best_model_path = os.path.join(self.path, 'pcc_' + str(round(score, 4)) +'_epoch_' +str(iter).zfill(3)+'.pth.tar')
		self.best_model_path = os.path.join(self.path, self.model_name)
		torch.save(model.state_dict(), self.best_model_path)


def logging(content, filename='logging.txt'):
	with open(filename, 'a') as f:
		f.write(content + '\n')
