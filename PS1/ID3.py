from node import Node
from parse import data
from collections import Counter
import math
import numpy as np
import matplotlib.pyplot as plt
import random



def preprocess(examples):
    '''
    This function returns examples without having any missing values ('?')
    Choose the most common value under the attribute

    Input: examples
    Output: examples
    '''
    attributes = getAttributes(examples)
    for attribute in attributes:
        valueList = []
        for example in examples:
            valueList.append(example[attribute])
        value = Counter(valueList).most_common(1)[0][0]
        for example in examples:
            if example[attribute] == '?':
                example[attribute] = value
    #print(examples[0])
    return examples

def getBestAttribute(examples):
    '''
    This function returns the best attribute
    '''

    # Get all attributes from data set
    attributes = getAttributes(examples)

    infoGain = []
    for attribute in attributes:
        # get the unique values for each attribute
        values = getValues(examples, attribute)
        # split the each value under each attribute
        totalVals = splitValues(examples, attribute, values)
        # get the total entropy for each attribute
        totalAttributeEnt = []
        for eachList in totalVals:
            # get classification of examples
            classification = getClassification(eachList)
            # get the Y (label) probability
            yProb = []
            for c in classification:
                eachExample = []
                for each in eachList:
                    if each['Class'] == c:
                        eachExample.append(each)
                prob = float(len(eachExample)) / float(len(eachList))
                #print(prob)
                yProb.append(prob)
            # get the total entropy for each list in total list
            totalEnt = 0
            for prob in yProb:
                if prob != 0:
                    ent = getEntropy(prob)
                else:
                    ent = 0
                totalEnt = totalEnt + ent * (float(len(eachList)) / float(len(examples)))  # sum of P(xi = P) * H(Y|xi=P)
            totalAttributeEnt.append(totalEnt)
            #print(yProb)
        entSum = sum(totalAttributeEnt)
        infoGain.append(entSum)
    infoGain = np.array(infoGain)
    #print(infoGain)

    #bestAttribute = attributes[np.argmin(infoGain)]
    #return bestAttribute
    return infoGain

def chooseBestAttribute(examples, infoGain):
    attributes = getAttributes(examples)
    bestAttribute = attributes[np.argmin(infoGain)]
    return bestAttribute

def getAttributes(examples):
    '''
    This function returns a list including all attributes in the dataset

    Input: examples of dataset
    Output: attributes of the examples (unique)
    '''
    attributes = []
    for i in range(len(examples)):
        for j in range(len(examples[0]) - 1):
            attributes.append(examples[i].keys()[j])
    # print(len(np.unique(attributes)))
    return np.unique(attributes)

def getValues(examples, attribute):
    '''
    This function returns a list of values corresponding to each attribute

    Input: examples of dataset, attributes of examples
    Output: values corresponding to each attribute (unique)
    '''
    values = []
    for example in examples:
        values.append(example[attribute])
    return np.unique(values)

def splitValues(examples, attribute, values):
    '''
    This function returns several lists of splited values corresponding to each attribute

    Input: examples of dataset, attribute of examples, values of attribute
    Output: splited values corresponding to each attribute
    '''
    totalVals = []
    for value in values:
        eachVals = []
        for example in examples:
            if example[attribute] == value:
                eachVals.append(example)
        totalVals.append(eachVals)
    # print(totalVals)
    # print(len(totalVals))
    return totalVals

def getClassification(examples):
    '''
    This function returns a list of classification of all examples

    Input: examples of dataset
    Output: classification of examples (all, non-unique)
    '''
    classification = []
    for example in examples:
        classification.append(example['Class'])
    return np.unique(classification)

def getEntropy(prob):
    '''
    This function returns the entropty corresponding to each label

    Input: probility of each label (classification)
    Output: entropy of the label
    '''
    return ((-1) * prob * math.log(prob, 2))  # H(Y) = - P(Y = k) log P(Y = k)

def getMode(examples):
    '''
    This function returns the most common classification label

    Output: classification label
    '''
    classification = []
    for example in examples:
        classification.append(example['Class'])
    mode = Counter(classification).most_common(1)[0][0]
    return mode

def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    tree = Node()
    if len(examples) == 0:
        tree.label = default
    elif (len(getAttributes(examples)) == 0 or len(getClassification(examples)) == 1):
        tree.label = getMode(examples)
    else:
        infoGain = getBestAttribute(examples)
        if len(np.unique(infoGain)) == 1:
            #print(np.unique(infoGain))
            tree.label = getMode(examples)
        else:
            bestAttribute = chooseBestAttribute(examples, infoGain)
            #print(bestAttribute)
            tree.branch = bestAttribute
            values = getValues(examples, bestAttribute)
            for eachVal in values:
                subExamples = []
                for example in examples:
                    if example[bestAttribute] == eachVal:
                        subExamples.append(example)
                subtree = ID3(subExamples, getMode(subExamples))
                tree.children[eachVal] = subtree
    return tree


def prune(tree, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''



def test(tree, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    totalNum = len(examples)
    # print(totalNum)
    correctNum = 0
    for example in examples:
        correctClassification = example['Class']
        if evaluate(tree, example) == correctClassification:
            correctNum = correctNum + 1
    print "The accuracy is =", correctNum / float(totalNum)
    return correctNum / float(totalNum)


def evaluate(tree, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    while (tree.label == None):
        value = example[tree.branch]
        tree = tree.children[value]

    return tree.label


def main():
    # preprocess the dataset so that dataset does not have any missing value
    examples = preprocess(data)
    # randomly shuffle the dataset
    random.shuffle(examples)
    # roughly choose around 300 training examples, around 80 testing examples, and around 40 validation examples
    size = len(examples)
    thresh1, thresh2 = int(0.7 * size), int(0.9 * size)
    trainingList = examples[:thresh1]
    validationList = examples[thresh1: thresh2]
    testingList = examples[thresh2:]
    # ID3 algorithm
    tree = ID3(trainingList, '')
    # check accuracy
    test(tree, testingList)



if __name__ == "__main__": main()
