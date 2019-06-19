import numpy as np 
np.random.seed(1234)
import h5py as hp
import scipy.io as sio
import pdb
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

def loadDataH5(fileName):
	f = hp.File(fileName, 'r')
	allKeys=[k for k in f.keys()]
	data = []
	for i in range(len(allKeys)):
		d = f[allKeys[i]][:]
		data.append(d)
	f.close()
	return data

def getAWA2Data(data):
	for i in range(len(data)):
		# pdb.set_trace()
		if len(data[i].shape) == 1:
			labels = data[i]
		elif data[i].shape[1] == 2048:
			features = data[i]
		elif data[i].shape[1] == 85:
			attributes = data[i]
		elif data[i].shape[1] == 300:
			word2vecs = data[i]
		elif data[i].shape[1] == 1:
			labels = data[i]
		else:
			raise NotImplementedError
	return [features, attributes, word2vecs, labels]

def getGtEmbeddings(data):
	for i in range(len(data)):
		if len(data[i].shape) == 1:
			labels = data[i]
		elif data[i].shape[1] == 85:
			attributes = data[i]
		elif data[i].shape[1] == 300:
			word2vecs = data[i]
		elif data[i].shape[1] == 1:
			labels = data[i]
		else:
			raise NotImplementedError
	return [attributes, word2vecs, labels]

def shuffleInUnision(data):
	nMatrices = len(data)
	nSamples = data[0].shape[0]
	for i in range(len(data)):
		assert nSamples == data[i].shape[0]
	perm = np.random.permutation(nSamples)
	permutedData = []
	for i in range(len(data)):
		permutedData.append(data[i][perm])
	return permutedData


def getGtEmbeddingsSubset(labelSubset, wholeLabels, wholeEmbeddings):
	embSubset = []
	for i in range(labelSubset.shape[0]):
		ind = np.where(wholeLabels == labelSubset[i])
		embSubset.append(wholeEmbeddings[ind])
	embSubset = np.array(embSubset)
	return embSubset

def uniqueLabels(labels):
	labels = np.unique(labels)
	labels = np.reshape(labels, (labels.shape[0], ))
	return labels

def getOneHotFormat(labels, uniqueLabels):
	oneHotLabels = np.zeros((labels.shape[0], uniqueLabels.shape[0]))
	for i in range(oneHotLabels.shape[0]):
		ind = np.where(uniqueLabels == labels[i])
		oneHotLabels[i][ind] = 1.0
	return oneHotLabels

def getOneHot(Labels):
	uLabels = np.unique(Labels)
	oneHot = np.zeros((Labels.shape[0], uLabels.shape[0]))
	for i in range(uLabels.shape[0]):
		ind = np.where(Labels == uLabels[i])
		oneHot[i][ind] = 1.0
	return oneHot


def findKNN(prediction, vecs, metric='euclidean'):
	if metric == 'dot':
		distances = np.dot(prediction, np.transpose(vecs))
		preds = np.argsort(distances, axis=1)
		preds = preds[:,::-1]
	elif metric == 'cosine':
		distances = cdist(prediction, vecs, 'cosine')
		preds = np.argsort(distances, axis=1)
	elif metric == 'euclidean':
		distances = cdist(prediction, vecs, 'euclidean')
		preds = np.argsort(distances, axis=1)
	else:
		raise NotImplementedError
	return preds

def getAcc(data, gtData, predictions, dataSet, k = 1, setting = 'GZSL', model = 'W_A_I', mode='NN', plotConfusion=False):
	# pdb.set_trace()
	[features, attributes, word2vecs, labels] = getAWA2Data(data)
	[gtAttributes, gtWord2vecs, gtLabels] = getGtEmbeddings(gtData)
	if mode == 'NN':
		if model == 'W_A_I':
			preds = findKNN(features, predictions)
		elif model == 'I_A':
			preds = findKNN(predictions, gtAttributes)
		elif model == 'I_W':
			preds = findKNN(predictions, gtWord2vecs)
		elif model == 'W_I':
			preds = findKNN(features, predictions)
		else:
			raise NotImplementedError
	if mode == 'classifier':
		# pdb.set_trace()
		preds = np.argsort(predictions, axis=-1)
		preds = preds[:,::-1]
	if setting == 'GZSL':
		preds = preds[:,0:k]
		preds = gtLabels[preds]
		lab = gtLabels
		# confMatrix = np.zeros((gtLabels.shape[0], gtLabels.shape[0]))
		cnf_matrix = confusion_matrix(labels, preds, gtLabels)
	elif setting == 'ZSL':
		labelsToConsider = uniqueLabels(labels)
		labelsPredicted = gtLabels[preds]
		refinedPreds = []
		lab = labels
		confMatrix = np.zeros((labelsToConsider.shape[0], labelsToConsider.shape[0]))
		for i in range(preds.shape[0]):
			for j in range(preds.shape[1]):
				if labelsPredicted[i][j] in labelsToConsider:
					refinedPreds.append(labelsPredicted[i][j])
					break
		# pdb.set_trace()
		refinedPreds = np.array(refinedPreds)
		assert refinedPreds.shape[0] == labels.shape[0]
		labelsWillBeConsidered = uniqueLabels(refinedPreds)
		try:
			assert labelsWillBeConsidered.shape[0] <= labelsToConsider.shape[0]
		except:
			pdb.set_trace()
		preds = refinedPreds
		cnf_matrix = confusion_matrix(labels, preds, labelsToConsider)
	else:
		raise NotImplementedError
	acc = []
	classes = uniqueLabels(labels)
	nClasses = classes.shape[0]
	# pdb.set_trace()
	for i in range(nClasses):
		curLab = classes[i]
		subset = labels == curLab
		predSubset = preds[subset][:]
		labSubset = labels[subset][:]
		predSubset = np.reshape(predSubset, (predSubset.shape[0]))
		labSubset = np.reshape(labSubset, (labSubset.shape[0]))
		curAcc = np.sum(predSubset == labSubset)/float(labSubset.shape[0])
		acc.append(curAcc)
		if setting == 'GZSL':
			pass
			# pdb.set_trace()
	acc = np.array(acc)
	acc = np.mean(acc)
	if plotConfusion:
		sio.savemat(setting+'_'+dataSet+'_confMat.mat', {'cnf_matrix':cnf_matrix})
		# print("Test Labels are")
	# print(model+" "+setting+" "+dataSet+" acc is "+ str(acc))
	return acc


