import glob
import os
import random
import numpy as np
from model import DFRNet
from utils import label

def weight(path, dtype, name, shape, arr=None):
    f = open(path, mode='w')
    if(len(shape)==1):
        if(arr!=None):
            W = np.load(arr)
        else:
            # W = np.random.randn(shape[0])
            W = np.zeros(100).astype(np.int64)
        f.write(f"{dtype} {name}[{shape[0]}] = {{")
        for i in range(shape[0]-1):
            f.write(f'{W[i]}, ')
        f.write(f'{W[shape[0]-1]}')
        f.write('};\n')
    elif(len(shape)==2):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randn(shape[0],shape[1])
        f.write(f"{dtype} {name}[{shape[0]}][{shape[1]}] = {{\n")
        for i in range(shape[0]-1):
            f.write('{')
            for j in range(shape[1]-1):
                f.write(f'{W[i][j]}, ')
            f.write(f'{W[i][shape[1]-1]}}},\n')
        f.write('{')
        for j in range(W.shape[1]-1):
            f.write(f'{W[shape[0]-1][j]}, ')
        f.write(f'{W[shape[0]-1][shape[1]-1]}}}\n')
        f.write('};\n')
    elif(len(shape)==3):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randn(shape[0],shape[1],shape[2])
        f.write(f"{dtype} {name}[{shape[0]}][{shape[1]}][{shape[2]}] = {{\n")
        for i in range(shape[0]-1):
            f.write('{')
            for j in range(shape[1]-1):
                f.write('{')
                for k in range(shape[2]-1):   
                    f.write(f'{W[i][j][k]}, ')
                f.write(f'{W[i][j][shape[2]-1]}}},')
            f.write('{')
            for k in range(shape[2]-1):   
                f.write(f'{W[i][shape[1]-1][k]}, ')
            f.write(f'{W[i][shape[1]-1][shape[2]-1]}}}}},\n')
        f.write('{')
        for j in range(shape[1]-1):
            f.write('{')
            for k in range(shape[2]-1):   
                f.write(f'{W[shape[0]-1][j][k]}, ')
            f.write(f'{W[shape[0]-1][j][shape[2]-1]}}},')
        f.write('{')
        for k in range(shape[2]-1):   
            f.write(f'{W[shape[0]-1][shape[1]-1][k]}, ')
        f.write(f'{W[shape[0]-1][shape[1]-1][shape[2]-1]}}}}}\n')
        f.write('};\n')
    elif(len(shape)==4):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randn(shape[0],shape[1],shape[2],shape[3])
        f.write(f"{dtype} {name}[{shape[0]}][{shape[1]}][{shape[2]}][{shape[3]}] = {{\n")
        for i in range(shape[0]-1):
            f.write('{')
            for j in range(shape[1]-1):
                f.write('{')
                for k in range(shape[2]-1):
                    f.write('{')
                    for l in range(shape[3]-1):
                        f.write(f'{W[i][j][k][l]}, ')
                    f.write(f'{W[i][j][k][shape[3]-1]}}},')
                f.write('{')
                for l in range(shape[3]-1):   
                    f.write(f'{W[i][j][shape[2]-1][l]}, ')
                f.write(f'{W[i][j][shape[2]-1][shape[3]-1]}}}}},\n')
            f.write('{')
            for k in range(shape[2]-1):
                f.write('{')
                for l in range(shape[3]-1):
                    f.write(f'{W[i][shape[1]-1][k][l]}, ')
                f.write(f'{W[i][shape[1]-1][k][shape[3]-1]}}},')
            f.write('{')
            for l in range(shape[3]-1):   
                f.write(f'{W[i][shape[1]-1][shape[2]-1][l]}, ')
            f.write(f'{W[i][shape[1]-1][shape[2]-1][shape[3]-1]}}}}}}},\n')
        f.write('{')
        for j in range(shape[1]-1):
            f.write('{')
            for k in range(shape[2]-1):
                f.write('{')
                for l in range(shape[3]-1):
                    f.write(f'{W[shape[0]-1][j][k][l]}, ')
                f.write(f'{W[shape[0]-1][j][k][shape[3]-1]}}},')
            f.write('{')
            for l in range(shape[3]-1):   
                f.write(f'{W[shape[0]-1][j][shape[2]-1][l]}, ')
            f.write(f'{W[shape[0]-1][j][shape[2]-1][shape[3]-1]}}}}},\n')
        f.write('{')
        for k in range(shape[2]-1):
            f.write('{')
            for l in range(shape[3]-1):
                f.write(f'{W[shape[0]-1][shape[1]-1][k][l]}, ')
            f.write(f'{W[shape[0]-1][shape[1]-1][k][shape[3]-1]}}},')
        f.write('{')
        for l in range(shape[3]-1):   
            f.write(f'{W[shape[0]-1][shape[1]-1][shape[2]-1][l]}, ')
        f.write(f'{W[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1]}}}}}}}\n')
        f.write('};\n')
    else:
        raise ValueError('shape is unvalid')
    f.close()
    
if __name__ == "__main__":
    try:
        os.mkdir('HLS')
    except:
        pass
    
    files = sorted(glob.glob('./dataset_append/*'), key=str.lower)
    for file in files:
        datasetname = os.path.splitext(os.path.basename(file))[0]
        print(datasetname)
        
        try:
            os.mkdir(f'HLS/{datasetname}')
        except:
            pass
        try:
            os.mkdir(f'HLS/{datasetname}/{datasetname}')
        except:
            pass
        
        resrep = 'DPRR'
        path = f'dataset_append/{datasetname}.npz'
        dataset = np.load(path)
        train_U = dataset['X'] # ex)ARAB:(6600,93,13+1)
        train_D = dataset['Y'] # ex)ARAB:(6600,10)
        test_U = dataset['Xte'] # ex)ARAB:(2200,93,13+1)
        test_D = dataset['Yte'] # ex)ARAB:(2200,10)
        
        k = 1/max(np.max(train_U[:,:,0:train_U.shape[2]-1]), np.max(test_U[:,:,0:test_U.shape[2]-1]))
        train_U = train_U*k
        test_U = test_U*k
        
        for i in range(train_U.shape[0]):
            np.savetxt(f'HLS/{datasetname}/{datasetname}/TRAIN_{i}.csv', train_U[i], delimiter=',')
        for i in range(test_U.shape[0]):
            np.savetxt(f'HLS/{datasetname}/{datasetname}/TEST_{i}.csv', test_U[i], delimiter=',')
        
        N_u = train_U.shape[2]-1
        N_x = 10
        N_y = train_D.shape[1]
        model = DFRNet(N_u,N_x,N_y,resrep,0.01,0.01)
        
        idx_arr = np.zeros((25,len(train_U)),dtype='int')
        for epoch in range(25):
            random.seed(epoch)
            idx_arr[epoch] = random.sample(range(len(train_U)), len(train_U))
        
        np.save(f'HLS/{datasetname}/win', model.layers['input'].params[0])
        np.save(f'HLS/{datasetname}/train_d', label(train_D))
        np.save(f'HLS/{datasetname}/test_d', label(test_D))
        np.save(f'HLS/{datasetname}/idx', idx_arr.reshape(-1))
        
        weight(f'HLS/{datasetname}/win.h', 'float', 'Win', np.load(f'HLS/{datasetname}/win.npy').shape, f'HLS/{datasetname}/win.npy')
        weight(f'HLS/{datasetname}/train_d.h', 'int', 'train_D', [len(np.load(f'HLS/{datasetname}/train_d.npy'))], f'HLS/{datasetname}/train_d.npy')
        weight(f'HLS/{datasetname}/test_d.h', 'int', 'test_D', [len(np.load(f'HLS/{datasetname}/test_d.npy'))], f'HLS/{datasetname}/test_d.npy')
        weight(f'HLS/{datasetname}/idx.h', 'int', 'idx_arr', np.load(f'HLS/{datasetname}/idx.npy').shape, f'HLS/{datasetname}/idx.npy')
        