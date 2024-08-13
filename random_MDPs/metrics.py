import numpy as np
import scipy.stats as st



def TV(X,Y):
    tv=0.5*np.linalg.norm(np.array(X)-np.array(Y),ord=1)
    return tv

def KL(X,Y):
    out=0
    for i in range(len(X)):
        x,y=X[i,:],Y[i,:]
        out=out+st.entropy(x,qk=y)
    return out

def KL_policy(X, Y):
    out = 0
    for i in range(len(X)):
        x, y = X[i, :], Y[i, :]
        out = out + st.entropy(x, qk=y)
    return out


def W(X,Y):
    out=0
    for i in range(len(X)):
        x,y=X[i,:],Y[i,:]
        out=out+my_ws(x, y)
    return out

def my_ws(X_in,Y_in):
    X=X_in[:]
    Y=Y_in[:]
    out=0
    movee=0
    for index in range(len(X)-1):
        #print(X,Y)
        movee=max(X[index],Y[index])-min(X[index],Y[index])
        out=out+movee
        if X[index]>Y[index]:
            X[index+1]=X[index+1]+movee
            X[index]=X[index]-movee
        elif Y[index]>X[index]:
            Y[index+1]=Y[index+1]+movee
            Y[index]=Y[index]-movee
    #print(X,Y)
    return out

def Lip(X):
    Ns, Na, Ns = X.shape
    grad_list = []
    for s1 in range(Ns):
        for a in range(Na):
            for s2 in range(Ns):
                if s2 != s1:
                    grad = W(X[s1, a, :][None, ...], X[s2, a, :][None, ...]) / np.abs(
                        s1 - s2
                    )
                    grad_list.append(grad)
    return max(grad_list)

def Lip_R(R):
    Ns = R.shape[0]
    grad_list = []
    for s1 in range(Ns):
            for s2 in range(s1+1, Ns):
                grad = np.abs(R[s1] - R[s2]) / np.abs(s1 - s2)
                grad_list.append(grad)
    return max(grad_list)
