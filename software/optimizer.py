import copy
import numpy as np

class MultiStepLR:
    def __init__(self, milestones, gamma=0.1):
        self.milestones = milestones # list
        self.gamma = gamma
        self.iter = 0
        self.coeff = 1
    
    def __call__(self):
        return self.coeff
    
    def step(self):
        self.iter += 1
        if self.iter in self.milestones:
            self.coeff = self.coeff * self.gamma

class pos_SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads, scheduler=None):
        params_copy = copy.deepcopy(params)
        for i in range(len(params)):
            if(scheduler is None):
                params_copy[i] -= self.lr * grads[i]
            else:
                params_copy[i] -= scheduler() * self.lr * grads[i]
        if((0 <= np.array(params_copy)).all() and (np.array(params_copy) < 0.5).all()):
            return params_copy
        else:
            return params

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads, scheduler=None):
        for i in range(len(params)):
            if(scheduler is None):
                params[i] -= self.lr * grads[i]
            else:
                params[i] -= scheduler() * self.lr * grads[i]
        
        return params
            
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads, scheduler=None):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            if(scheduler is None):
                self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            else:
                self.v[i] = self.momentum * self.v[i] - scheduler() * self.lr * grads[i]
            params[i] += self.v[i]
        
        return params

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads, scheduler=None):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            if(scheduler is None):
                params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)
            else:
                params[i] -= scheduler() * self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)
        
        return params

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads, scheduler=None):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        if(scheduler is None):
            lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        else:
            lr_t = scheduler() * self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
        
        return params