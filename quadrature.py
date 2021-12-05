from scipy import integrate as integrate
import numpy as np
import pickle

sigma = 7./32

def f(r):
    return 2*r**3 - 3*r**2 +1

def squared_norm(x1, x2, y1, y2):
    return np.power(x1 -y1, 2) + np.power(x2 -y2, 2)

def kappa(x1, x2, y1, y2):
    if squared_norm(x1,x2,y1,y2)<=8:
        return 1.
    elif squared_norm(x1,x2,y1,y2) <= 25./2:
        return f((squared_norm(x1, x2, y1,y2)-8)/(9./2))
    else:
        return 0.

def int_jk(x1, x2, y1, y2,sigma):
    return np.power(squared_norm(x1,x2,y1,y2),-1-1*sigma)*kappa(x1,x2,y1,y2)

def tilde_h2(x,y,z, sigma):
    return np.power(np.power(x+y,2) + np.power(z,2), -1-sigma)

def hat_h2(x,y,z, sigma):
    return -1.0*z*np.power(np.power(x+y,2)+np.power(z,2),-1-sigma)

def h3(x1,x2,y1,y2, sigma):
    return np.power(np.power(x1+y1,2) + np.power(x2+y2,2),-1-sigma)

def int_21(x,y,sigma):
    return tilde_h2(1,x,y,sigma) + tilde_h2(x,1,y,sigma) + tilde_h2(x,y,1,sigma)

def int_22(x,y,sigma):
    return hat_h2(1,x,y,sigma) + hat_h2(x,1,y,sigma) + hat_h2(x,y,1,sigma)

def int_3(x, y, z, sigma):
    return h3(1,x,y,z, sigma)

sig = sigma
ints = {}
print('compute quadrature for ', 2)
ints["int2_1"] = integrate.nquad(int_21, [[0,1],[0,1]], args=(sig,))
ints["int2_2"] = integrate.nquad(int_22, [[0,1],[0,1]], args=(sig,))
print('compute quadrature for ', 3)
ints["int3"] = integrate.nquad(int_3, [[0,1],[0,1],[0,1]], args=(sig,))

indexes = [[2,0], [2,1], [2,2], [3,0], [3,1], [3,2], [3,3], [4,0], [4,1], [4,2]]
for k in range(len(indexes)):
    print('compute quadrature for ', k+4)
    ints["int"+str(4+k)] = integrate.nquad(int_jk,
                                           [[0,1], [0,1],
                                            [indexes[k][0], indexes[k][0]+1], [indexes[k][1], indexes[k][1]+1]],
                                           args=(sig,))


string = "integral_approximations_sigma_" + str(sigma) + ".pkl"
file = open(string, "wb")
pickle.dump(ints, file)
file.close()

file = open(string, "rb")
output = pickle.load(file)
print(output)