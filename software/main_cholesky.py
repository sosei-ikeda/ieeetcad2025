import csv
import glob
import os
import time
import numpy as np
from model import DFRNet, CholeskyTrainer
from utils import label, ACC


if __name__ == "__main__":
    resrep = 'DPRR'

    ABbeta = np.loadtxt('./result/date_best.csv', delimiter=',', skiprows=1, usecols=[1,2,3])

    with open('./result/cholesky.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dataset','ACC'])
    
    files = sorted(glob.glob('./dataset_append/*'), key=str.lower)
    for i,file in enumerate(files):
        datasetname = os.path.splitext(os.path.basename(file))[0]
        A,B,beta = ABbeta[i]
        print(datasetname)

        path = f'dataset_append/{datasetname}.npz'
        dataset = np.load(path)
        train_U = dataset['X'] # ex)ARAB:(6600,93,13+1)
        train_D = dataset['Y'] # ex)ARAB:(6600,10)
        test_U = dataset['Xte'] # ex)ARAB:(2200,93,13+1)
        test_D = dataset['Yte'] # ex)ARAB:(2200,10)
        
        k = 1/max(np.max(train_U[:,:,0:train_U.shape[2]-1]), np.max(test_U[:,:,0:test_U.shape[2]-1]))
        train_U = train_U*k
        test_U = test_U*k
        
        N_u = train_U.shape[2]-1
        N_x = 10
        N_y = train_D.shape[1]

        model = DFRNet(N_u,N_x,N_y,resrep,A,B)

        trainer = CholeskyTrainer(model, N_x, N_y, resrep, 10**beta*k**2)
        
        try:
            trainer.fit(train_U, train_D)
        except:
            pass
        
        Y_pred = []
        for i in range(len(test_U)):
            y_pred = model.predict(test_U[i])
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        
        test_pred = np.empty(0,int)
        for i in range(len(Y_pred)):
            test_pred = np.hstack((test_pred, np.argmax(Y_pred[i]))) 
    
        test_label = label(test_D)
        test_ACC = ACC(test_pred, test_label)            
            
        end = time.time()
        
        print(f'ACC:{test_ACC}')   
            
                
        with open('./result/cholesky.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datasetname, test_ACC])