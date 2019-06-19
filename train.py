import utils
import Model
import torch
from torch import nn
from torch.utils.data.data_utils import TensorDataset
from torch.utils.data import DataLoader
from toch.autograd import Variable
import numpy
import argparse
import os
import time

dataset = 'AWA2'
path = '../../rawDatasets/zsl_summarized/data/' + dataset + '/'
save_path = 'saved_model'

data = utils.loadDataH5(path + 'trainValLoc.h5')
[features_train, attrVecs_train, wordVecs_train, labels_train] = utils.getAWA2Data(data)

data = utils.loadDataH5(path + 'testSeenLoc.h5')
[features_test, attrVecs_test, wordVecs_test, labels_test] = utils.getAWA2Data(data)

data = utils.loadDataH5(path + 'testUnseenLoc.h5')
[features_testU, attrVecs_testU, wordVecs_testU, labels_testU] = utils.getAWA2Data(data)

data = utils.loadDataH5(path + 'embeddings.h5')
[attrVecs, word2Vecs, Labels] = utils.getGtEmbeddings(data)

[features_train, attrVecs_train, wordVecs_train, labels_train] = utils.shuffleInUnision(
    [features_train, attrVecs_train, wordVecs_train, labels_train])


batchSize = 64

num_epochs = 200
dataset_tr = TensorDataset(features_train, attrVecs_train, wordVecs_train, labels_train)
loader_tr = DataLoader(dataset_tr, batchSize)

dataset_ts = TensorDataset(features_test, attrVecs_test, wordVecs_test, labels_test)
loader_ts = DataLoader(dataset_ts, batchSize)


labTrain = utils.uniqueLabels(labels_train)
wordVecTr = utils.getGtEmbeddingsSubset(labels_train, labTrain, word2Vecs)
attrTr = utils.getGtEmbeddingsSubset(labels_train, labTrain, attrVecs)

labsT = utils.getOneHotFormat(labels_train, labTrain)


def divide_batches(train_data, n=64):

    for i in range(0, len(train_data), n):
        yield train_data[i:i+n]

def findKNN(prediction, vecs):
    distances = cdist(prediction, vecs, 'cosine')
    preds = numpy.argsort(distances, axis=1)
    return preds


def trainW2A(learning_rate):
    model = Model.W2AModel(300, 85).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in loader_tr:
            _, word_vec, attr_vec, _ = data
            input = Variable(torch.from_numpy(numpy.array(word_vec)))
            target = Variable(torch.from_numpy(numpy.array(attr_vec)))
            attr_pred = model(input)
            loss = criterion(attr_pred, target)

            # zero the gradient
            optimizer.zero_grad()

            # perform a backward pass
            loss.backward()

            # update the parameters
            optimizer.update()

        if((num_epochs % 50) == 0):
            # save the model (work on this: find out how to save)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
            }, save_path)

    # save the final model
    torch.save(model, save_path+'/W2AModel.pth')


def testW2A():

    # restore model and use it to predict the attributes for the test classes
    model = torch.load(save_path + '/W2AModel.pth')
    model.eval()

    for data in loader_ts:
        _, word_vecs, attr_vecs, label = data
        word2Vecs = Variable(torch.from_numpy(numpy.array(word_vecs)))
        target = Variable(torch.from_numpy(numpy.array(attr_vecs)))
        attr_pred = model(word2Vecs)



def trainAE(learning_rate, loss_lambda):

    # create a model of Autoencoder
    model = Model.AEModel(85, 2048).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in loader_tr:
            feature_vec, _, attr_vec, _ = data
            input = Variable(torch.from_numpy(numpy.array(attr_vec)))
            target_h = Variable(torch.from_numpy(numpy.array(feature_vec)))
            target = Variable(torch.from_numpy(numpy.array(attr_vec)))

            output_h, output = model(input)
            loss = criterion(target_h, output_h) + (loss_lambda*criterion(input, output))

            # zero the gradient
            optimizer.zero_grad()

            # perform a backward pass
            loss.backward()

            # update the parameters
            optimizer.update()

        if ((num_epochs % 50) == 0):
            # save the model (work on this: find out how to save)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_path)

            # save the final model
        torch.save(model, save_path + '/AEModel.pth')


def testAEModel():

    model = torch.load(save_path+'AEModel.pth')
    model.eval()

    attr_vecs = Variable(torch.from_numpy(numpy.array(attrVecs_test)))
    pred_embedding, _ = model(attr_vecs)
    preds = findKNN(pred_embedding, features_test)
    acc = []

    classes = numpy.unique(labels_test)
    nClasses = classes.shape[0]

    for i in range(nClasses):
        subset = labels_test == classes[i]
        subset = numpy.reshape(subset, (subset.shape[0], ))
        predSubset = preds[subset][:]
        label_subset = labels_test[subset]
        counter = 0
        for j in range(predSubset.shape[0]):
            curPreds = predSubset[j]
            if label_subset[j] in curPreds:
                counter = counter + 1
        curAcc = counter/float(predSubset.shape[0])
        acc.append(curAcc)

    acc = numpy.array(acc)
    acc = numpy.mean(acc)

    print("ZSL Accuracy is =" + str(acc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=string, default='AWA2', help='Datset to train and test the code on')
    parser.add_argument('--dest_path', type=string, default='results', help='destination folder to store results')
    parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate for Adam')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    timestr = time.strftime("%d-%m-%Y-%H:%M:%S")
    args.dest_path = args.dest_path.join(timestr)
    save_path = os.path.dirname(os.path.abspath(__file__)).join(args.dest_path)
    learning_rate = args.lr
    dataset = args.dataset

    if not os.path.exits(save_path):
        os.makedirs(save_path)

    trainAE(learning_rate=learning_rate, loss_lambda=0.01)

    testAEModel()









