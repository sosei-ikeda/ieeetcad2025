import copy
import numpy as np

class Mask:
    def __init__(self, W):
        self.params = [W]
        
    def forward(self, u):
        W, = self.params
        return np.dot(u,W)
    
    def backward(self, dout):
        return None
    
class identityDFR:
    def __init__(self, N_x, A, B):
        self.params = [A,B]
        self.grads = [np.zeros_like(A),np.zeros_like(B)]
        self.x_prevprev = np.zeros(N_x)
        self.x_prev = np.zeros(N_x)
        self.J = np.zeros(N_x)
        self.N_x = N_x
    
    def forward(self, J):
        A,B = self.params
        X = np.zeros(self.N_x)
        X[0] = float(A)*(J[0]+self.x_prev[0])+float(B)*self.x_prev[self.N_x-1]
        for i in range(self.N_x-1):
            X[i+1] = float(A)*(J[i+1]+self.x_prev[i+1])+float(B)*X[i]
        self.x_prevprev = copy.deepcopy(self.x_prev)
        self.x_prev = X
        self.J = J
        return self.x_prev
    
    def backward(self, dout):
        A, B = self.params
        dX = np.zeros(self.N_x)
        dA = 0
        dB = 0
        for i in range(self.N_x):
            dX[self.N_x-1-i] += dout[self.N_x-1-i]
            if(self.N_x-1-i != 0):
                dX[self.N_x-1-i-1] += float(B)*dX[self.N_x-1-i]
                dB += self.x_prev[self.N_x-1-i-1]*dX[self.N_x-1-i]
            else:
                dB += self.x_prevprev[self.N_x-1]*dX[self.N_x-1-i]
            dA += (self.J[self.N_x-1-i]+self.x_prevprev[self.N_x-1-i])*dX[self.N_x-1-i]
        self.grads[0] = dA
        self.grads[1] = dB
        return None
    
    def refresh(self):
        self.x_prev = np.zeros(self.N_x)

class MGDFR:
    def __init__(self, N_x, A, B):
        self.params = [A,B]
        self.grads = [np.zeros_like(A),np.zeros_like(B)]
        self.x_prevprev = np.zeros(N_x)
        self.x_prev = np.zeros(N_x)
        self.J = np.zeros(N_x)
        self.N_x = N_x
    
    def f(self, i):
        return i/(1+i**2)
    
    def forward(self, J):
        A,B = self.params
        X = np.zeros(self.N_x)
        X[0] = float(A)*self.f(J[0]+self.x_prev[0])+float(B)*self.x_prev[self.N_x-1]
        for i in range(self.N_x-1):
            X[i+1] = float(A)*self.f(J[i+1]+self.x_prev[i+1])+float(B)*X[i]
        self.x_prevprev = copy.deepcopy(self.x_prev)
        self.x_prev = X
        self.J = J
        return self.x_prev
    
    def backward(self, dout):
        A, B = self.params
        dX = np.zeros(self.N_x)
        dA = 0
        dB = 0
        for i in range(self.N_x):
            dX[self.N_x-1-i] += dout[self.N_x-1-i]
            if(self.N_x-1-i != 0):
                dX[self.N_x-1-i-1] += float(B)*dX[self.N_x-1-i]
                dB += self.x_prev[self.N_x-1-i-1]*dX[self.N_x-1-i]
            else:
                dB += self.x_prevprev[self.N_x-1]*dX[self.N_x-1-i]
            dA += self.f(self.J[self.N_x-1-i]+self.x_prevprev[self.N_x-1-i])*dX[self.N_x-1-i]
        self.grads[0] = dA
        self.grads[1] = dB
        return None
    
    def refresh(self):
        self.x_prev = np.zeros(self.N_x)

class LRS:
    def __init__(self, N_x):
        self.N_x = N_x
        self.r = np.zeros((N_x))
        self.T = 1
    
    def forward(self, x):
        self.r = x
    
    def backward(self, dout):
        # the shape of dout: N_r
        return dout
    
    def refresh(self,T):
        self.r = np.zeros((self.N_x))

class DPRR:
    def __init__(self, N_x):
        self.N_x = N_x
        self.r = np.zeros((N_x,N_x+1))
        self.x_prev = None
        self.T = 0
    
    def forward(self, x, x_prev):
        self.x_prev = copy.deepcopy(x_prev[0:self.N_x])
        x = np.reshape(x, (-1,1))
        x_prev = np.reshape(x_prev, (-1,1))
        self.r += np.dot(x, x_prev.T)
    
    def backward(self, dout):
        # the shape of dout: N_r
        return (self.x_prev*np.sum(dout[0:self.N_x*self.N_x].reshape((self.N_x,self.N_x)),axis=1) \
                +dout[self.N_x*self.N_x:])
    
    def refresh(self,T):
        self.T = T
        self.r = np.zeros((self.N_x,self.N_x+1))

class Output:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.r = None
    
    def forward(self, r):
        W,b = self.params
        self.r = r.reshape(1,-1)
        return np.dot(W,r)+b
    
    def backward(self, dout):
        # the shape of dout: N_y
        W, b = self.params
        dr = np.dot(W.T, dout)
        dout = dout.reshape(-1,1)
        dW = np.dot(dout, self.r)
        # db = np.sum(dout)
        db = dout

        self.grads[0] = dW
        self.grads[1] = db.reshape(-1)
        # return dr.reshape(-1)*b.shape
        return dr.reshape(-1)

def softmax(x):
    x = x - np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    return x

def cross_entropy_error(y, t):
    t = t.argmax()
    return -np.sum(np.log(y[t] + 1e-7))

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.d = None

    def forward(self, y, d):
        self.y = softmax(y)
        self.d = d
        loss = cross_entropy_error(self.y, self.d)
        return loss

    def backward(self, dout=1):
        dx = self.y - self.d
        return dx

class Tikhonov:
    def __init__(self, N_r, N_y, beta):
        self.beta = beta
        self.R_RT = np.zeros((N_r, N_r))
        self.D_RT = np.zeros((N_y, N_r))
        self.I = np.identity(N_r)
        
    def __call__(self, d, r):
        r = np.reshape(r, (-1, 1)).astype(np.float64)
        d = np.reshape(d, (-1, 1))
        self.R_RT += np.dot(r, r.T)
        self.D_RT += np.dot(d, r.T)
    
    def get_Wout_opt(self):
        R_pseudo_inv = np.linalg.inv(self.R_RT + self.beta*self.I)
        Wout_opt = np.dot(self.D_RT, R_pseudo_inv)
        # print(R_pseudo_inv)
        # raise RuntimeError
        return Wout_opt

class Cholesky:
    def __init__(self, N_r, N_y, beta):
        self.beta = beta
        self.R_RT = np.zeros((N_r, N_r))
        self.D_RT = np.zeros((N_y, N_r))
        self.I = np.identity(N_r)
        
    def __call__(self, d, r):
        r = np.reshape(r, (-1, 1)).astype(np.float64)
        d = np.reshape(d, (-1, 1))
        self.R_RT += np.dot(r, r.T)
        self.D_RT += np.dot(d, r.T)
    
    def get_Wout_opt(self):
        B = self.R_RT + self.beta*self.I
        
        # cholesky decomposition of B; B=CC^T
        n,_ = np.shape(B)
        C = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                num = B[i][j]
                if(i>j):
                    k = j
                    while(k>0):
                        k -= 1
                        num -= C[i][k]*C[j][k]
                    C[i][j] = num/C[j][j]
                elif(i==j):
                    k = j
                    while(k>0):
                        k -= 1
                        num -= C[i][k]**2
                    C[i][j] = np.sqrt(num)
                else:
                    C[i][j] = 0
        
        # calculate X=(WoutC) in case (WoutC)C^T=D_RT
        m,_ = np.shape(self.D_RT)
        X = np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                num = self.D_RT[i][j]
                k = j
                while(k>0):
                    k -= 1
                    num -= X[i][k]*C[j][k]
                X[i][j] = num/C[j][j]
        
        # calculate Wout in case WoutC=X
        Wout_opt = np.zeros([m,n])
        for i in range(m):
            for j in reversed(range(n)):
                num = X[i][j]
                k = n
                while(k>j):
                    k -= 1
                    num -= Wout_opt[i][k]*C[k][j]
                    
                Wout_opt[i][j] = num/C[j][j]

        return Wout_opt