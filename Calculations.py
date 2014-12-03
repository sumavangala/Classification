__author__ = 'suma'


#Accuracy, Error Rate, Sensitivity, Specificity, Precision, F-1 Score, Fbeat score (beta = 0.5 and 2)
#8 metrics * 2 methods * 4datasets * 2 (training and test)
#input these numbers
TP = 393
FN = 66
FP = 178
TN = 41

Accuracy = (TP+TN)*1.0/(TP+FN+FP+TN)
ErrorRate = (FP+FN)*1.0/(TP+FN+FP+TN)
Sensitivity = TP*1.0/(TP+FN)
Specificity = TN*1.0/(FP+TN)
Precision = TP*1.0/(TP+FP)
Recall = TP*1.0/(TP+FN)
F1Score = (2.0*Precision*Recall)/(Precision+Recall)
FBeta1 = (1.25*Precision*Recall)/((0.25*Precision)+Recall)
FBeat2 = (5.0*Precision*Recall)/((4*Precision)+Recall)

print("%f %f %f %f %f %f %f %f" % (Accuracy, ErrorRate, Sensitivity, Specificity, Precision, F1Score, FBeta1, FBeat2))