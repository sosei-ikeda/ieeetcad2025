import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from model import DFRNet, RidgeTrainer
from utils import label, ACC


if __name__ == "__main2__":
    resrep = 'DPRR'
    

    datasetname = 'CHAR'
    beta = 1e-4
    
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
    
    
    f = open('./result/grid_CHAR_1.csv', 'w')
    f.write('A,B,ACC\n')
    f.close()
    
    for A in [0.03,3]:
        for B in [0.03,3]:
            print(A,B,beta)

            model = DFRNet(N_u,N_x,N_y,resrep,A,B)
        
            trainer = RidgeTrainer(model, N_x, N_y, resrep, beta)
            
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
            
            f = open('./result/grid_CHAR_1.csv', 'a')
            f.write(f'{A},{B},{test_ACC}\n')
            f.close()
     
    f = open('./result/grid_CHAR_2.csv', 'w')
    f.write('A,B,ACC\n')
    f.close()
    
    for A in [0.01,0.1,1,10]:
        for B in [0.01,0.1,1,10]:
            print(A,B,beta)
        
            model = DFRNet(N_u,N_x,N_y,resrep,A,B)
        
            trainer = RidgeTrainer(model, N_x, N_y, resrep, beta)
            
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
            
            print(f'ACC:{test_ACC}')   
            
            f = open('./result/grid_CHAR_2.csv', 'a')
            f.write(f'{A},{B},{test_ACC}\n')
            f.close()
     
if __name__ == "__main__":
    data1 = pd.read_csv('./result/grid_CHAR_1.csv')
    df_pivot1 = pd.pivot_table(data=data1, values='ACC', 
                              columns='A', index='B', aggfunc=np.mean)
    data2 = pd.read_csv('./result/grid_CHAR_2.csv')
    df_pivot2 = pd.pivot_table(data=data2, values='ACC', 
                              columns='A', index='B', aggfunc=np.mean)
    
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(1,2,1)
    sn.heatmap(df_pivot1, annot=True, fmt='.2f',
                vmin=0.5, vmax=1,
               cmap='binary', cbar=False, xticklabels=False, yticklabels=False)
    ax1.invert_yaxis()
    plt.xlabel('$log_{10} p$')
    plt.ylabel('$log_{10} q$')
    
    ax2 = fig.add_subplot(1,2,2)
    sn.heatmap(df_pivot2, annot=True, fmt='.2f',
                vmin=0.5, vmax=1,
               cmap='binary', cbar=False, xticklabels=False, yticklabels=False)
    ax2.invert_yaxis()
    plt.xlabel('$log_{10} p$')
    plt.ylabel('$log_{10} q$')
    
    plt.savefig('./result/grid_CHAR.svg', bbox_inches='tight', pad_inches=0) 