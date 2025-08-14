import csv
import glob
import os
import time
import math
import numpy as np
from model import DFRNet, RidgeTrainer
from utils import label, ACC

def n_divide(n,range_list):
    p,q = range_list
    div_list = []
    for i in range(n):
        div_list += [((2*n-(2*i+1))*p+(2*i+1)*q)/(2*n)]
    return div_list

if __name__ == "__main__":
    resrep = 'DPRR'

    accs = np.loadtxt('./result/gd.csv', delimiter=',', skiprows=1, usecols=[4])
    with open('./result/gs.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dataset','divs','ACC','time','total_time'])
    
    files = sorted(glob.glob('./dataset_append/*'), key=str.lower)
    for i,file in enumerate(files):
        datasetname = os.path.splitext(os.path.basename(file))[0]
        acc = accs[i]
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
        N_x = 30
        N_y = train_D.shape[1]
        
        A_range = [-3.75,-0.25]
        B_range = [-2.75,-0.25]

        ACC_list = []
        time_list = []
        n = 0

        while(1):
            n += 1
            print(f'division {n}')
            best_loss = 100

            A_list = n_divide(n,A_range)
            B_list = n_divide(n,B_range)
            
            start = time.time()
            for beta in [1e-6,1e-4,1e-2,1e+0]:
                for A in A_list:
                    for B in B_list:
                        model = DFRNet(N_u,N_x,N_y,resrep,10**A,10**B)

                        if(resrep=='DPRR'):
                            trainer = RidgeTrainer(model, N_x, N_y, resrep, 10**beta*k**2)
                        else:
                            trainer = RidgeTrainer(model, N_x, N_y, resrep, 10**beta*k**2)
                        
                        try:
                            trainer.fit(train_U, train_D)
                        except:
                            pass

                        total_loss = 0
                        for i in range(len(train_U)):
                            loss = model.forward(train_U[i], train_D[i])
                            total_loss += loss
                        train_loss = total_loss / len(train_U)
                        if(np.round(train_loss,2-math.floor(math.log10(abs(train_loss)))-1) <= np.round(best_loss,2-math.floor(math.log10(abs(best_loss)))-1)):
                            best_loss = train_loss
                            best_beta = beta
                            best_param = [A,B]
        
            model = DFRNet(N_u,N_x,N_y,resrep,10**best_param[0],10**best_param[1])
            if(resrep=='DPRR'):
                trainer = RidgeTrainer(model, N_x, N_y, resrep, 10**best_beta*k**2)
            else:
                trainer = RidgeTrainer(model, N_x, N_y, resrep, 10**best_beta*k**2)
        
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
            print(f'time:{end-start}s')   
            
            print(f'best_ACC:{test_ACC}')   
            print(f'best_param:{best_param}')
            print(f'best_beta:{best_beta}\n')
            
            ACC_list += [test_ACC]
            time_list += [end-start]
            
            if(np.round(acc,3)<=np.round(test_ACC,3)):
                break
            if(n>=2):
                if(test_ACC<=ACC_list[-2] and np.round(acc,2)<=np.round(test_ACC,2)):
                    break
                
        with open('./result/gs.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datasetname, n, ACC_list, time_list, sum(time_list)])