__author__ = 'suma'


import numpy
import math
import sys


def main():
    k = 5 #number of weak classifiers
    trainFileContent = {}
    totalAttributes = 0
    totalRecords = 0
    attributes = []  #1 to totalAttributes
    weights = {}
    sampleSet = {}
    totalPositiveClassRecords = {}
    totalNegativeClassRecords = {}
    positiveClassDict = [dict() for x in range(k)]
    negativeClassDict = [dict() for x in range(k)]
    errorRate = {}

    #Find the total number of attributes - max from train and test file
    trainFile = open(train_file_path, 'r')
    testFile = open(test_file_path, 'r')
    index = 1
    for line in trainFile:
        trainFileContent[index] = line.rstrip('\n')
        index += 1
    for key in trainFileContent:
        line = trainFileContent[key].strip()
        if len(line) > 0:
            totalRecords = totalRecords + 1
            record = line.split()
            values = record[len(record)-1].split(':')
            if int(values[0]) > totalAttributes:
                totalAttributes = int(values[0])
    for line in testFile:
        line = line.strip()
        if len(line) > 0:
            record = line.split()
            values = record[len(record)-1].split(':')
            if int(values[0]) > totalAttributes:
                totalAttributes = int(values[0])

    for x in range(1, k+1, 1):
        totalPositiveClassRecords[x] = 0
        totalNegativeClassRecords[x] = 0

    for x in range(1, totalAttributes + 1, 1):
        attributes.append(x)

    for x in range(1, totalRecords + 1, 1):
        weights[x] = 1*1.0/totalRecords
        sampleSet[x] = x

    #run the classifier k rounds
    for counter in range(1, k+1, 1):
        sample = numpy.random.choice(sampleSet.values(), totalRecords, 1, p=weights.values())

        for each in sample:
            line = trainFileContent[each]
            record = line.split()
            if record[0] == '+1':
                totalPositiveClassRecords[counter] += 1
                record.pop(0)
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))
                    if(not positiveClassDict[counter-1].has_key(r)):
                        positiveClassDict[counter-1][r] = 1
                    else:
                        count = positiveClassDict[counter-1][r]
                        count += 1
                        positiveClassDict[counter-1][r] = count

                diff = set(attributes) - set(attrList) #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(not positiveClassDict[counter-1].has_key(key)):
                        positiveClassDict[counter-1][key] = 1
                    else:
                        count = positiveClassDict[counter-1][key]
                        count = count + 1
                        positiveClassDict[counter-1][key] = count
            else:
                totalNegativeClassRecords[counter] += 1
                record.pop(0)
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))
                    if(not negativeClassDict[counter-1].has_key(r)):
                        negativeClassDict[counter-1][r] = 1
                    else:
                        count = negativeClassDict[counter-1][r]
                        count = count + 1
                        negativeClassDict[counter-1][r] = count

                diff = set(attributes) - set(attrList) #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(not negativeClassDict[counter-1].has_key(key)):
                        negativeClassDict[counter-1][key] = 1
                    else:
                        count = negativeClassDict[counter-1][key]
                        count = count + 1
                        negativeClassDict[counter-1][key] = count

        #probability of +1
        Pr1 = (totalPositiveClassRecords[counter]*1.0)/totalRecords
        #propability of -1
        Pr2 = (totalNegativeClassRecords[counter]*1.0)/totalRecords

        actualLabels = {}
        predictedLabels = {}
        index = 0
        #for each in trainFileContent:
        for each in sample:
            line = trainFileContent[each]
            #line = trainFileContent[each].strip()
            if len(line) > 0:
                index += 1
                record = line.split()
                actualLabels[index] = record[0]
                record.pop(0)
                positiveTotalProb = Pr1
                negativeTotalProb = Pr2
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))

                    if(positiveClassDict[counter-1].has_key(r)):
                        positiveTotalProb = positiveTotalProb * ((positiveClassDict[counter-1][r]*1.0)/totalPositiveClassRecords[counter])
                    else:
                        positiveTotalProb = positiveTotalProb * 0

                    if(negativeClassDict[counter-1].has_key(r)):
                        negativeTotalProb = negativeTotalProb * ((negativeClassDict[counter-1][r]*1.0)/totalNegativeClassRecords[counter])
                    else:
                        negativeTotalProb = negativeTotalProb * 0

                diff = set(attributes) - set(attrList)  #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(positiveClassDict[counter-1].has_key(key)):
                        positiveTotalProb = positiveTotalProb * ((positiveClassDict[counter-1][key]*1.0)/totalPositiveClassRecords[counter])
                    else:
                        positiveTotalProb = positiveTotalProb * 0

                    if(negativeClassDict[counter-1].has_key(key)):
                        negativeTotalProb = negativeTotalProb * ((negativeClassDict[counter-1][key]*1.0)/totalNegativeClassRecords[counter])
                    else:
                        negativeTotalProb = negativeTotalProb * 0

                if positiveTotalProb > negativeTotalProb:
                    predictedLabels[index] = '+1'
                else:
                    predictedLabels[index] = '-1'

        #Calculating the weighted error rate
        error = 0
        newWeightSum = 0
        oldWeightSum = 0
        for key in actualLabels:
            actualClass = actualLabels[key]
            predictedClass = predictedLabels[key]
            if (actualClass == '+1' and predictedClass == '-1') or (actualClass == '-1' and predictedClass == '+1'):
                error = error + weights[key]
        errorRate[counter] = error

        if error < 0.5:
            #Adjusting the weights of a tuple if correctly classified
            for key in actualLabels:
                actualClass = actualLabels[key]
                predictedClass = predictedLabels[key]
                oldWeightSum = oldWeightSum + weights[key]
                if (actualClass == '+1' and predictedClass == '+1') or (actualClass == '-1' and predictedClass == '-1'):
                    weights[key] = weights[key] * ((error*1.0)/(1-error))
                newWeightSum = newWeightSum + weights[key]

            #normalize the weights
            if(newWeightSum != 0):
                for key in actualLabels:
                    weights[key] = weights[key] * float(oldWeightSum)/newWeightSum

    #Predict labels for training data and print FP,TP,TN,FN
    index = 0
    actualLabels = {}
    predictedLabels = {}

    for each in trainFileContent:
        line = trainFileContent[each].strip()
        if len(line) > 0:
            index = index + 1
            record = line.split()
            actualLabels[index] = record[0]
            record.pop(0)

            #Combine k classifiers to classify the dataset
            for counter in range(1, k+1, 1):
                result = 0

                #probability of +1
                Pr1 = (totalPositiveClassRecords[counter]*1.0)/totalRecords
                #propability of -1
                Pr2 = (totalNegativeClassRecords[counter]*1.0)/totalRecords

                positiveTotalProb = Pr1
                negativeTotalProb = Pr2
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))

                    if(positiveClassDict[counter-1].has_key(r)):
                        positiveTotalProb = positiveTotalProb * ((positiveClassDict[counter-1][r]*1.0)/totalPositiveClassRecords[counter])
                    else:
                        positiveTotalProb = positiveTotalProb * 0

                    if(negativeClassDict[counter-1].has_key(r)):
                        negativeTotalProb = negativeTotalProb * ((negativeClassDict[counter-1][r]*1.0)/totalNegativeClassRecords[counter])
                    else:
                        negativeTotalProb = negativeTotalProb * 0

                diff = set(attributes) - set(attrList)  #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(positiveClassDict[counter-1].has_key(key)):
                        positiveTotalProb = positiveTotalProb * ((positiveClassDict[counter-1][key]*1.0)/totalPositiveClassRecords[counter])
                    else:
                        positiveTotalProb = positiveTotalProb * 0

                    if(negativeClassDict[counter-1].has_key(key)):
                        negativeTotalProb = negativeTotalProb * ((negativeClassDict[counter-1][key]*1.0)/totalNegativeClassRecords[counter])
                    else:
                        negativeTotalProb = negativeTotalProb * 0

                if(errorRate[counter] != 0):
                    if positiveTotalProb > negativeTotalProb:
                        result = result + math.log(2, float(1-errorRate[counter])/errorRate[counter])
                    else:
                        result = result - math.log(2, float(1-errorRate[counter])/errorRate[counter])

            if result > 0:
                predictedLabels[index] = '+1'
            else:
                predictedLabels[index] = '-1'


    #Calcualate TP, FN, FP, TN for training dataset
    truePositives = 0
    falseNegatives = 0
    falsePositives = 0
    trueNegatives = 0

    for key in actualLabels:
        actualClass = actualLabels[key]
        predictedClass = predictedLabels[key]
        if(actualClass == '+1' and predictedClass == '+1'):
            truePositives += 1
        elif(actualClass == '+1' and predictedClass == '-1'):
            falseNegatives += 1
        elif(actualClass == '-1' and predictedClass == '+1'):
            falsePositives += 1
        elif(actualClass == '-1' and predictedClass == '-1'):
            trueNegatives += 1

    print "%d %d %d %d" % (truePositives, falseNegatives, falsePositives, trueNegatives)

    #Predict labels for test data and print FP,TP,TN,FN
    index = 0
    actualLabels = {}
    predictedLabels = {}

    testFile = open(test_file_path, 'r')
    for line in testFile:
        line = line.strip()
        if len(line) > 0:
            index = index + 1
            record = line.split()
            actualLabels[index] = record[0]
            record.pop(0)

            #Combine k classifiers to classify the dataset
            for counter in range(1, k+1, 1):
                result = 0

                #probability of +1
                Pr1 = (totalPositiveClassRecords[counter]*1.0)/totalRecords
                #propability of -1
                Pr2 = (totalNegativeClassRecords[counter]*1.0)/totalRecords

                positiveTotalProb = Pr1
                negativeTotalProb = Pr2
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))

                    if(positiveClassDict[counter-1].has_key(r)):
                        positiveTotalProb = positiveTotalProb * ((positiveClassDict[counter-1][r]*1.0)/totalPositiveClassRecords[counter])
                    else:
                        positiveTotalProb = positiveTotalProb * 0

                    if(negativeClassDict[counter-1].has_key(r)):
                        negativeTotalProb = negativeTotalProb * ((negativeClassDict[counter-1][r]*1.0)/totalNegativeClassRecords[counter])
                    else:
                        negativeTotalProb = negativeTotalProb * 0

                diff = set(attributes) - set(attrList)  #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(positiveClassDict[counter-1].has_key(key)):
                        positiveTotalProb = positiveTotalProb * ((positiveClassDict[counter-1][key]*1.0)/totalPositiveClassRecords[counter])
                    else:
                        positiveTotalProb = positiveTotalProb * 0

                    if(negativeClassDict[counter-1].has_key(key)):
                        negativeTotalProb = negativeTotalProb * ((negativeClassDict[counter-1][key]*1.0)/totalNegativeClassRecords[counter])
                    else:
                        negativeTotalProb = negativeTotalProb * 0

                if(errorRate[counter] != 0):
                    if positiveTotalProb > negativeTotalProb:
                        result = result + math.log((1-errorRate[counter])*1.0/errorRate[counter])
                    else:
                        result = result - math.log((1-errorRate[counter])*1.0/errorRate[counter])

            if result > 0:
                predictedLabels[index] = '+1'
            else:
                predictedLabels[index] = '-1'


    #Calcualate TP, FN, FP, TN for training dataset
    truePositives = 0
    falseNegatives = 0
    falsePositives = 0
    trueNegatives = 0

    for key in actualLabels:
        actualClass = actualLabels[key]
        predictedClass = predictedLabels[key]
        if(actualClass == '+1' and predictedClass == '+1'):
            truePositives += 1
        elif(actualClass == '+1' and predictedClass == '-1'):
            falseNegatives += 1
        elif(actualClass == '-1' and predictedClass == '+1'):
            falsePositives += 1
        elif(actualClass == '-1' and predictedClass == '-1'):
            trueNegatives += 1

    print "%d %d %d %d" % (truePositives, falseNegatives, falsePositives, trueNegatives)


if __name__ == "__main__":
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    main()