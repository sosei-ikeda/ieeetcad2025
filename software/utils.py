import numpy as np

def label(Y):
    Label = np.empty(0,int)
    for i in range(len(Y)):
        Label = np.hstack((Label, np.argmax(Y[i]))) 
    return Label

def ACC(Label1, Label2):
    count = 0
    for i in range(len(Label1)):
        if Label1[i] == Label2[i]:
            count += 1
    return(count/len(Label1))