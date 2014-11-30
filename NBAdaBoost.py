__author__ = 'suma'


import numpy


def main():
    k = 5
    trainFileContent = {}
    totalAttributes = 0
    totalRecords = 0
    attributes = []  #1 to totalAttributes
    weights = [dict() for x in range(k+1)]
    sampleSet = [dict() for x in range(k+1)]
    totalPositiveClassRecords = {}
    totalNegativeClassRecords = {}
    positiveClassDict = [dict() for x in range(k)]
    negativeClassDict = [dict() for x in range(k)]
    # actualLabels = [dict() for x in range(k)]
    # predictedLabels = [dict() for x in range(k)]

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
        weights[0][x] = 1*1.0/totalRecords
        sampleSet[0][x] = x

    #run the classifier k rounds
    for counter in range(1, k+1, 1):
        sample = numpy.random.choice(sampleSet[counter-1].values(), totalRecords, p=weights[counter-1].values())
        index = 1
        for x in sample:
            sampleSet[counter][index] = x
            index += 1

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
        for each in trainFileContent:
            line = trainFileContent[each].strip()
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

        break
    # print totalNegativeClassRecords
    # print totalPositiveClassRecords
    # print positiveClassDict
    # print negativeClassDict


if __name__ == "__main__":
    train_file_path = "/home/suma/Desktop/Data Mining/Assignment/Assignment4/dataset/adult.train"
    test_file_path = "/home/suma/Desktop/Data Mining/Assignment/Assignment4/dataset/adult.test"
    main()