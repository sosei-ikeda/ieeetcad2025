import copy
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from layer import *

class DFRNet:
    def __init__(self, N_u, N_x, N_y, resrep, A_init=1e-16, B_init=0.82, seed=0):
        np.random.seed(seed)
        self.N_x = N_x
        Win = (-1)**np.random.randint(0,2,(N_u,N_x))
        A = A_init
        B = B_init
        if(resrep=='LRS'):
            Wout = 0. * np.random.randn(N_y,N_x)
        elif(resrep=='DPRR'):
            Wout = 0. * np.random.randn(N_y,N_x*(N_x+1))
        else:
            raise ValueError
        bout = np.zeros(N_y)

        if(resrep=='LRS'):
            self.layers = {'input': Mask(Win), 'reservoir': identityDFR(N_x,A,B), 
                           'resrep': LRS(N_x), 'output': Output(Wout, bout)}
        elif(resrep=='DPRR'):
            self.layers = {'input': Mask(Win), 'reservoir': identityDFR(N_x,A,B), 
                            'resrep': DPRR(N_x), 'output': Output(Wout, bout)}
            
        self.loss_layer = SoftmaxWithLoss()
        self.resrep = resrep
    
    def r(self, u):
        self.layers['reservoir'].refresh()
        self.layers['resrep'].refresh(T=u.shape[0])
        x_prev = np.zeros(self.N_x)
        j = 0
        while(1):
            x = u[j,:u.shape[1]-1]
            x = self.layers['input'].forward(x)
            x = self.layers['reservoir'].forward(x)
            
            if(self.resrep=='LRS'):
                self.layers['resrep'].forward(x)
            elif(self.resrep=='DPRR'):
                self.layers['resrep'].forward(x, np.append(x_prev,1))
                
            x_prev = x
                
            if(u[j][u.shape[1]-1]!=0 or j==u.shape[0]-1):
                break
            
            j += 1
        
        return (self.layers['resrep'].r).reshape(-1)

    def predict(self, u):
        x = self.layers['output'].forward(self.r(u))
        return x

    def forward(self, u, d):
        y = self.predict(u)
        loss = self.loss_layer.forward(y, d)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = self.layers[layer].backward(dout)

class GradTrainer:
    def __init__(self, model, optimizer_reservoir, scheduler_reservoir, optimizer_output, scheduler_output):
        self.model = model
        self.optimizer_reservoir = optimizer_reservoir
        self.scheduler_reservoir = scheduler_reservoir
        self.optimizer_output = optimizer_output
        self.scheduler_output = scheduler_output
        self.loss_list = []
        self.eval_interval = None
        self.param_list = [copy.deepcopy(model.layers['reservoir'].params)]

    def fit(self, U, D, epochs=30, seed=0, skip=True):
        model, optimizer_reservoir, scheduler_reservoir, optimizer_output, scheduler_output = self.model, self.optimizer_reservoir, self.scheduler_reservoir, self.optimizer_output, self.scheduler_output
        best_param = None
        best_loss = 100
        total_loss = 0
        loss_count = 0
        
        for epoch in range(epochs):
            random.seed(seed+epoch)
            idx_list = random.sample(range(len(U)), len(U))

            start_time = time.time()
            for i in range(len(U)):
                num = idx_list[i]
                loss = model.forward(U[num], D[num])
                model.backward()
                        
                total_loss += loss
                loss_count += 1
                
                model.layers['reservoir'].params = \
                    copy.deepcopy(optimizer_reservoir.update(model.layers['reservoir'].params, \
                                  model.layers['reservoir'].grads, \
                                  scheduler_reservoir))
                model.layers['output'].params = \
                    copy.deepcopy(optimizer_output.update(model.layers['output'].params, \
                                            model.layers['output'].grads, \
                                            scheduler_output))
            
                self.param_list.append(copy.deepcopy(model.layers['reservoir'].params))
            
            avg_loss = total_loss / loss_count
            elapsed_time = time.time() - start_time
            print('| epoch %d / %d | time %d[s] | loss %.5f'
                  % (epoch+1, epochs, elapsed_time, avg_loss))
            self.loss_list.append(float(avg_loss))
            if(best_loss > avg_loss and scheduler_reservoir()!=1):
                best_loss = avg_loss
                best_param = copy.deepcopy(model.layers['reservoir'].params)
            total_loss, loss_count = 0, 0
            if scheduler_reservoir is not None:
                scheduler_reservoir.step()
            if scheduler_output is not None:
                scheduler_output.step()
            
            if(skip==True):
                if(avg_loss < 1.0):
                    return
        
        model.layers['reservoir'].params = copy.deepcopy(best_param)

    def plot(self, file, ylim=None):
        plt.figure()
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(file, bbox_inches='tight', pad_inches=0)  
    
    def param_plot(self, file):
        plt.figure(figsize=(8,8))
        xy = np.array(self.param_list)
        plt.plot(xy[:,0],xy[:,1],'o--')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.savefig(file, bbox_inches='tight', pad_inches=0)

class RidgeTrainer:
    def __init__(self, model, N_x, N_y, resrep, beta):
        self.model = model
        if(resrep=='LRS'):
            self.N_r = N_x
        elif(resrep=='DPRR'):
            self.N_r = N_x*(N_x+1)
        self.optimizer = Tikhonov(self.N_r+1, N_y, beta)

    def fit(self, U, D):
        for i in range(len(U)):
            self.optimizer(D[i], np.append(self.model.r(U[i]),1))
        
        W = self.optimizer.get_Wout_opt()[:,0:self.N_r]
        b = self.optimizer.get_Wout_opt()[:,self.N_r].reshape(-1)
        self.model.layers['output'].params = [W,b]

class CholeskyTrainer:
    def __init__(self, model, N_x, N_y, resrep, beta):
        self.model = model
        if(resrep=='LRS'):
            self.N_r = N_x
        elif(resrep=='DPRR'):
            self.N_r = N_x*(N_x+1)
        self.optimizer = Cholesky(self.N_r+1, N_y, beta)

    def fit(self, U, D):
        for i in range(len(U)):
            self.optimizer(D[i], np.append(self.model.r(U[i]),1))
        
        W = self.optimizer.get_Wout_opt()[:,0:self.N_r]
        b = self.optimizer.get_Wout_opt()[:,self.N_r].reshape(-1)
        self.model.layers['output'].params = [W,b]