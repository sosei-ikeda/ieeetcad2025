import numpy as np

def cholesky(X):
    row, column = np.shape(X)
    if(row != column):
        raise ValueError
    else:
        n = row
    Y = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            num = X[i][j]
            if(i>j):
                k = j
                while(k>0):
                    k -= 1
                    num -= Y[i][k]*Y[j][k]
                Y[i][j] = num/Y[j][j]
            elif(i==j):
                k = j
                while(k>0):
                    k -= 1
                    num -= Y[i][k]**2
                Y[i][j] = np.sqrt(num)
            else:
                Y[i][j] = 0
    return Y

def f(C,D_RT):
    m,n = np.shape(D_RT)
    X = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            num = D_RT[i][j]
            k = j
            while(k>0):
                k -= 1
                num -= X[i][k]*C[j][k]
            X[i][j] = num/C[j][j]    
    return X

def g(C,X):
    m,n = np.shape(X)
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

# B = np.array([[4,8,4,6],[8,25,23,18],[4,23,38,28],[6,18,28,54]])
# C = cholesky(B)
# print(C)

C = np.array([[2,0,0,0],[4,3,0,0],[2,5,3,0],[3,2,4,5]])
X = np.array([[28,33,20,10],[30,22,19,20]])
print(g(C,X))