import h5py as hp
import numpy as np
from scipy import io as sio
from gensim.models import KeyedVectors
import pdb
from numpy import loadtxt


class DataLoader_AWA2:
    path = ""
    dataset = ""

    def __init__(self):
        pass

    def getAttrs(self):
        attrSplit = sio.loadmat(self.path + 'att_splits.mat')
        att = attrSplit['original_att']
        att = np.transpose(att)
        return att

    def getLabels(self):
        res101 = sio.loadmat(self.path + 'res101.mat')
        data = res101['labels']
        return data

    def getFeatures(self):
        res101 = sio.loadmat(self.path + 'res101.mat')
        data = res101['features']
        data = np.transpose(data)
        return data

    def getLabelNumberAssociation(self):
        attrSplit = sio.loadmat(self.path + 'att_splits.mat')
        allclasses_names = attrSplit['allclasses_names']
        labelNumberAssociation = []
        # pdb.set_trace()
        for i in range(allclasses_names.shape[0]):
            labelNumberAssociation.append([i + 1, allclasses_names[i][0][0]])
        return labelNumberAssociation

    def getLabelsFromTxt(self, file):
        text_file = open(file, "r")
        lines = text_file.readlines()
        text_file.close()
        data = []
        for i in range(len(lines)):
            data.append(lines[i][:-1])
        return data


    def getWord2VecsGlove(self, file):
        gloveFile = './../../preTrainedModels/glove.840B.300d.txt'
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done."+len(model)+" words loaded!")
        #pdb.set_trace()
        classes = self.getLabelsFromTxt(file)
        vecs = np.zeros((len(classes), 300))
        exclude = []
        for i in range(len(classes)):
            try:
                curClass = classes[i].strip()
                vecs[i] = model[curClass]
            except:
                # pdb.set_trace()
                # vecs[i] = np.ones((300,))*-100
                exclude.append(i+1)
                print('skipped the class: '+classes[i])
        return vecs, exclude

    def saveWordVecsAndAttrVecs(self):
        tosaveWordVecs = []
        tosaveAttrVecs = []
        Labels = []
        for i in range(self.word2VecsPerClass.shape[0]):
            if (i + 1) in self.exclude:
                pass
            else:
                tosaveWordVecs.append(self.word2VecsPerClass[i])
                tosaveAttrVecs.append(self.attrsPerClass[i])
                Labels.append(i + 1)
        tosaveWordVecs = np.array(tosaveWordVecs)
        tosaveAttrVecs = np.array(tosaveAttrVecs)
        Labels = np.array(Labels)
        # pdb.set_trace()
        fDst = hp.File(self.path + 'embeddings.h5', 'w')
        fDst.create_dataset("word2Vecs", data=tosaveWordVecs)
        fDst.create_dataset("attrVecs", data=tosaveAttrVecs)
        fDst.create_dataset("Labels", data=Labels)
        fDst.close()

    def saveWordVecsAndAttrVecsOrig(self):
        tosaveWordVecs = []
        tosaveAttrVecs = []
        Labels = []
        for i in range(self.word2VecsPerClass.shape[0]):
            tosaveWordVecs.append(self.word2VecsPerClass[i])
            tosaveAttrVecs.append(self.attrsPerClass[i])
            Labels.append(i + 1)
        tosaveWordVecs = np.array(tosaveWordVecs)
        tosaveAttrVecs = np.array(tosaveAttrVecs)
        Labels = np.array(Labels)
        # pdb.set_trace()
        fDst = hp.File(self.path + 'embeddingsOrig.h5', 'w')
        fDst.create_dataset("word2Vecs", data=tosaveWordVecs)
        fDst.create_dataset("attrVecs", data=tosaveAttrVecs)
        fDst.create_dataset("Labels", data=Labels)
        fDst.close()

    def makeTrainingData(self, file, featuresPerImage, labelNumberAssociation, word2VecsPerClass, exclude, labelsTrain1,
                         labelsPerImage, attrsPerClass):
        trainLabelNumbers = []
        for i in range(len(labelsTrain1)):
            for j in range(len(labelNumberAssociation)):
                if (labelsTrain1[i] == labelNumberAssociation[j][1]) and (labelNumberAssociation[j][0] not in exclude):
                    trainLabelNumbers.append(j + 1)
        consider = []
        wordVecs = []
        attrVecs = []
        labels = []
        for i in range(labelsPerImage.shape[0]):
            if labelsPerImage[i] in trainLabelNumbers:
                # pdb.set_trace()
                consider.append(1)
                wordVecs.append(word2VecsPerClass[labelsPerImage[i] - 1])
                attrVecs.append(attrsPerClass[labelsPerImage[i] - 1])
                labels.append(labelsPerImage[i])
            else:
                consider.append(0)
        consider = np.array(consider, dtype='bool')
        features = featuresPerImage[consider]
        attrVecs = np.reshape(np.array(attrVecs), (len(attrVecs), 85))
        wordVecs = np.reshape(np.array(wordVecs), (len(wordVecs), 300))
        labels = np.reshape(np.array(labels), (len(labels), 1))
        fDst = hp.File(self.path + file, 'w')
        fDst.create_dataset("features", data=features)
        fDst.create_dataset("attrVecs", data=attrVecs)
        fDst.create_dataset("wordVecs", data=wordVecs)
        fDst.create_dataset("labels", data=labels)
        fDst.close()
        print("Made " + file + " at " +self.path)

    # pdb.set_trace()

    def getData(self, A):
        feats = []
        labs = []
        w2v = []
        att = []
        A = np.reshape(A, (A.shape[0],))
        for i in range(A.shape[0]):
            curLabel = np.reshape(self.labelsPerImage[A[i]], (1,))
            if curLabel not in self.exclude:
                feats.append(self.featuresPerImage[A[i]])
                labs.append(curLabel)
                w2v.append(self.word2VecsPerClass[curLabel[0] - 1])
                att.append(self.attrsPerClass[curLabel[0] - 1])
        # feats.append(featuresPerImage[A[i]])
        # labs.append(curLabel)
        # w2v.append(word2VecsPerClass[curLabel[0]-1])
        # att.append(attrsPerClass[curLabel[0]-1])
        return (feats, labs, w2v, att)

    def saveToFile(self, file, data):
        (feats, labs, w2v, att) = (data[0], data[1], data[2], data[3])
        feats = np.reshape(np.array(feats), (len(feats), 2048))
        labs = np.reshape(np.array(labs), (len(labs),))
        w2v = np.reshape(np.array(w2v), (len(w2v), 300))
        att = np.reshape(np.array(att), (len(att), 85))
        assert (feats.shape[0] == labs.shape[0])
        assert (feats.shape[0] == w2v.shape[0])
        assert (feats.shape[0] == att.shape[0])
        print("Shape of " + file + ' is ' + str(feats.shape[0]))
        fDst = hp.File(self.path + file, 'w')
        fDst.create_dataset("features", data=feats)
        fDst.create_dataset("attrVecs", data=att)
        fDst.create_dataset("wordVecs", data=w2v)
        fDst.create_dataset("labels", data=labs)
        fDst.close()
        print("Made " + self.path + file)

    def makeLocData(self):
        mat = sio.loadmat(self.path + 'att_splits.mat')
        trainLoc = mat['train_loc'][:] - 1
        valLoc = mat['val_loc'][:] - 1
        trainValLoc = mat['trainval_loc'][:] - 1
        testSeenLoc = mat['test_seen_loc'][:] - 1
        testUnseenLoc = mat['test_unseen_loc'][:] - 1
        (feats, labs, w2v, att) = self.getData(trainLoc)
        self.saveToFile('trainLoc.h5', data=(feats, labs, w2v, att))
        (feats, labs, w2v, att) = self.getData(valLoc)
        self.saveToFile('valLoc.h5', data=(feats, labs, w2v, att))
        (feats, labs, w2v, att) = self.getData(trainValLoc)
        self.saveToFile('trainValLoc.h5', data=(feats, labs, w2v, att))
        (feats, labs, w2v, att) =self. getData(testSeenLoc)
        self.saveToFile('testSeenLoc.h5', data=(feats, labs, w2v, att))
        (feats, labs, w2v, att) = self.getData(testUnseenLoc)
        self.saveToFile('testUnseenLoc.h5', data=(feats, labs, w2v, att))

    # pdb.set_trace()
    # for i in range()
    # featuresPerImage
    # labelsPerImage
    # labelNumberAssociation
    # exclude
    # word2VecsPerClass
    # attrsPerClass
    def loadcsv(self, fileName):
        import csv
        names = []
        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                names.append(row)
        return names[0]

    def makeWord2VecsImageNet(self, fileName):
        # word2Vecs, e = getWord2Vecs(fileName)
        word2Vecs, e = self.getWord2VecsGlove(fileName)
        fDst = hp.File(self.path + 'class_label_Embeddings.h5', 'w')
        fDst.create_dataset("word2Vecs", data=word2Vecs)
        fDst.close()
        #pdb.set_trace()

    def makeData(self):

        labelNumberAssociation = self.getLabelNumberAssociation()
        labelsTrain1 = self.getLabelsFromTxt(file=self.path + 'trainclasses1.txt')
        labelsTrain2 = self.getLabelsFromTxt(file=self.path + 'trainclasses2.txt')
        labelsTrain3 = self.getLabelsFromTxt(file=self.path + 'trainclasses3.txt')
        labelsVal1 = self.getLabelsFromTxt(file=self.path + 'valclasses1.txt')
        labelsVal2 = self.getLabelsFromTxt(file=self.path + 'valclasses2.txt')
        labelsVal3 = self.getLabelsFromTxt(file=self.path + 'valclasses3.txt')
        labelsTest = self.getLabelsFromTxt(file=self.path + 'testclasses.txt')
        self.attrsPerClass = self.getAttrs()
        self.labelsPerImage = self.getLabels()
        self.featuresPerImage = self.getFeatures()
        # word2VecsPerClass, exclude = getWord2Vecs(path+'allClasses.txt')
        self.word2VecsPerClass, self.exclude = self.getWord2VecsGlove(self.path + 'allClasses.txt')
        self.saveWordVecsAndAttrVecs()
        self. makeTrainingData('train1.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass, self.exclude,
                         labelsTrain1, self.labelsPerImage, self.attrsPerClass)
        self.makeTrainingData('train2.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass, self.exclude,
                         labelsTrain2, self.labelsPerImage, self.attrsPerClass)
        self.makeTrainingData('train3.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass, self.exclude,
                         labelsTrain3, self.labelsPerImage, self.attrsPerClass)
        self. makeTrainingData('val1.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass, self.exclude, labelsVal1,
                         self.labelsPerImage, self.attrsPerClass)
        self.makeTrainingData('val2.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass,self. exclude, labelsVal2,
                         self.labelsPerImage, self.attrsPerClass)
        self.makeTrainingData('val3.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass,self. exclude, labelsVal3,
                         self.labelsPerImage, self.attrsPerClass)
        self.makeTrainingData('test.h5', self.featuresPerImage, labelNumberAssociation, self.word2VecsPerClass, self.exclude, labelsTest,
                         self.labelsPerImage, self.attrsPerClass)
        self.makeLocData()
        # makeWord2VecsImageNet('imagenet_categories.csv')
        self. makeWord2VecsImageNet('class_labels.csv')
    # saveWordVecsAndAttrVecsOrig()
    # pdb.set_trace()