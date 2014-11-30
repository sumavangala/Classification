__author__ = 'suma'


def main():
    positiveClassDict = {}
    negativeClassDict = {}
    totalRecords = 0
    totalPositiveClassRecords = 0
    totalNegativeClassRecords = 0
    totalAttributes = 0
    attributes = []  #1 to totalAttributes
    actualLabels = {}
    predictedLabels = {}

    #Find the total number of attributes - max from train and test file
    trainFile = open(train_file_path, 'r')
    testFile = open(test_file_path, 'r')
    for line in trainFile:
        line = line.strip()
        if len(line) > 0:
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

    for x in range(1, totalAttributes + 1, 1):
        attributes.append(x)


    #Finding the counts to calculate probability
    trainFile = open(train_file_path, 'r')
    for line in trainFile:
        line = line.strip()
        if len(line) > 0:
            record = line.split()
            totalRecords = totalRecords + 1
            actualLabels[totalRecords] = record[0]
            if record[0] == '+1':
                totalPositiveClassRecords = totalPositiveClassRecords + 1
                record.pop(0)
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))
                    if(not positiveClassDict.has_key(r)):
                        positiveClassDict[r] = 1
                    else:
                        count = positiveClassDict[r]
                        count = count + 1
                        positiveClassDict[r] = count

                diff = set(attributes) - set(attrList) #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(not positiveClassDict.has_key(key)):
                        positiveClassDict[key] = 1
                    else:
                        count = positiveClassDict[key]
                        count = count + 1
                        positiveClassDict[key] = count
            else:
                totalNegativeClassRecords = totalNegativeClassRecords + 1
                record.pop(0)
                attrList = []
                for r in record:
                    pair = r.split(':')
                    attrList.append(int(pair[0]))
                    if(not negativeClassDict.has_key(r)):
                        negativeClassDict[r] = 1
                    else:
                        count = negativeClassDict[r]
                        count = count + 1
                        negativeClassDict[r] = count

                diff = set(attributes) - set(attrList) #attributes for which values are 0
                for x in diff:
                    key = str(x) + ':0'
                    if(not negativeClassDict.has_key(key)):
                        negativeClassDict[key] = 1
                    else:
                        count = negativeClassDict[key]
                        count = count + 1
                        negativeClassDict[key] = count

    #probability of +1
    Pr1 = (totalPositiveClassRecords*1.0)/totalRecords
    #propability of -1
    Pr2 = (totalNegativeClassRecords*1.0)/totalRecords

    #Predicting labels for Training dataset
    trainFile = open(train_file_path, 'r')
    index = 0
    for line in trainFile:
        line = line.strip()
        if len(line) > 0:
            index = index + 1
            record = line.split()
            record.pop(0)
            positiveTotalProb = Pr1
            negativeTotalProb = Pr2
            attrList = []
            for r in record:
                pair = r.split(':')
                attrList.append(int(pair[0]))

                if(positiveClassDict.has_key(r)):
                    positiveTotalProb = positiveTotalProb * ((positiveClassDict[r]*1.0)/totalPositiveClassRecords)
                else:
                    positiveTotalProb = positiveTotalProb * 0

                if(negativeClassDict.has_key(r)):
                    negativeTotalProb = negativeTotalProb * ((negativeClassDict[r]*1.0)/totalNegativeClassRecords)
                else:
                    negativeTotalProb = negativeTotalProb * 0

            diff = set(attributes) - set(attrList)  #attributes for which values are 0
            for x in diff:
                key = str(x) + ':0'
                if(positiveClassDict.has_key(key)):
                    positiveTotalProb = positiveTotalProb * ((positiveClassDict[key]*1.0)/totalPositiveClassRecords)
                else:
                    positiveTotalProb = positiveTotalProb * 0

                if(negativeClassDict.has_key(key)):
                    negativeTotalProb = negativeTotalProb * ((negativeClassDict[key]*1.0)/totalNegativeClassRecords)
                else:
                    negativeTotalProb = negativeTotalProb * 0

            if positiveTotalProb > negativeTotalProb:
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
    print ((truePositives + trueNegatives)*100.0) / (truePositives+trueNegatives+falsePositives+falseNegatives)


    #Predicting labels for Test dataset
    testFile = open(test_file_path, 'r')
    index = 0
    actualLabels = {}
    predictedLabels = {}
    for line in testFile:
        line = line.strip()
        if len(line) > 0:
            index = index + 1
            record = line.split()
            actualLabels[index] = record[0]
            record.pop(0)
            positiveTotalProb = Pr1
            negativeTotalProb = Pr2
            attrList = []
            for r in record:
                pair = r.split(':')
                attrList.append(int(pair[0]))

                if(positiveClassDict.has_key(r)):
                    positiveTotalProb = positiveTotalProb * ((positiveClassDict[r]*1.0)/totalPositiveClassRecords)
                else:
                    positiveTotalProb = positiveTotalProb * 0

                if(negativeClassDict.has_key(r)):
                    negativeTotalProb = negativeTotalProb * ((negativeClassDict[r]*1.0)/totalNegativeClassRecords)
                else:
                    negativeTotalProb = negativeTotalProb * 0

            diff = set(attributes) - set(attrList)  #attributes for which values are 0
            for x in diff:
                key = str(x) + ':0'
                if(positiveClassDict.has_key(key)):
                    positiveTotalProb = positiveTotalProb * ((positiveClassDict[key]*1.0)/totalPositiveClassRecords)
                else:
                    positiveTotalProb = positiveTotalProb * 0

                if(negativeClassDict.has_key(key)):
                    negativeTotalProb = negativeTotalProb * ((negativeClassDict[key]*1.0)/totalNegativeClassRecords)
                else:
                    negativeTotalProb = negativeTotalProb * 0

            if positiveTotalProb > negativeTotalProb:
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
    print ((truePositives + trueNegatives)*100.0) / (truePositives+trueNegatives+falsePositives+falseNegatives)

if __name__ == "__main__":
    train_file_path = "/home/suma/Desktop/Data Mining/Assignment/Assignment4/dataset/adult.train"
    test_file_path = "/home/suma/Desktop/Data Mining/Assignment/Assignment4/dataset/adult.test"
    main()