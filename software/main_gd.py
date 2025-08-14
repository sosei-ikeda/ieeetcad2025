import csv
import glob
import os
import time
import math
import numpy as np
from model import DFRNet, GradTrainer, RidgeTrainer
from optimizer import MultiStepLR, pos_SGD, SGD
from utils import label, ACC

if __name__ == "__main__":
    resrep = 'DPRR'
    
    try:
        os.mkdir('./result')
    except:
        pass
    with open('./result/gd.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dataset','A','B','beta','ACC','time'])
    
    files = sorted(glob.glob('./dataset_append/*'), key=str.lower)
    for file in files:
        datasetname = os.path.splitext(os.path.basename(file))[0]
                
        print(datasetname)
        
        try:
            os.mkdir(f'./result/{datasetname}')
        except:
            pass
        
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
        
        model = DFRNet(N_u,N_x,N_y,resrep,0.01,0.01)
        
        start = time.time()
        
        scheduler_reservoir = MultiStepLR([5,10,15,20])
        optimizer_reservoir = pos_SGD(1e+0)
        scheduler_output = MultiStepLR([10,15,20])
        optimizer_output = SGD(lr=1e+0)
        trainer = GradTrainer(model, optimizer_reservoir, scheduler_reservoir, optimizer_output, scheduler_output)
        trainer.fit(train_U, train_D, 25)
        
        print('optimized [A,B]:',model.layers['reservoir'].params)
        
        N_x = 30
        model = DFRNet(N_u,N_x,N_y,resrep,model.layers['reservoir'].params[0],model.layers['reservoir'].params[1])
        
        best_beta = None
        best_ACC = 0
        best_loss = 100
        
        for beta in [1e-6,1e-4,1e-2,1e+0]:
            print(f'@beta:{beta}')
            trainer = RidgeTrainer(model, N_x, N_y, resrep, beta*k**2)
            try:
                trainer.fit(train_U, train_D)
            except:
                continue
            
            total_loss = 0
            for i in range(len(train_U)):
                loss = model.forward(train_U[i], train_D[i])
                total_loss += loss
            train_loss = total_loss / len(train_U)
            print(f'train_loss={train_loss}')
            if(np.round(train_loss,2-math.floor(math.log10(abs(train_loss)))-1) <= np.round(best_loss,2-math.floor(math.log10(abs(best_loss)))-1)):
                best_loss = train_loss
                best_beta = beta
            
        print(f'optimized beta:{best_beta}')
        
        trainer = RidgeTrainer(model, N_x, N_y, resrep, best_beta*k**2)
        trainer.fit(train_U, train_D)
        
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
        print(f'test_ACC={test_ACC}\n')
        
        end = time.time()
        print(f'time:{end-start}s')
        
        with open('./result/gd.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datasetname,model.layers['reservoir'].params[0],model.layers['reservoir'].params[1],best_beta,test_ACC,end-start])
