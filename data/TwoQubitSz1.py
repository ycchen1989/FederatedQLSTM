
import matplotlib.pyplot as plt
import numpy as np
###############################################################
Jval=1. # 0.1 0.5 1.0 2.0
##
def mysqrt(x):
    if x>=0:
        return np.sqrt(x)
    else:
        return 1.0j*np.sqrt(-x)
def sz1(t):
    return (1/(16*Jval**2-1))*np.exp(-t/2)*( (1-16*Jval**2)*np.exp(t/2) - 2*mysqrt(1-16*Jval**2)*np.sinh(0.5*mysqrt(1-16*Jval**2)*t) - 2*np.cosh(0.5*mysqrt(1-16*Jval**2)*t) + 32*Jval**2 )
##
t_total=100 # select a proper range
dt=0.001
time=np.linspace(0,t_total,int(t_total/dt)+1)
#
plt.figure()
plt.plot(time,sz1(time),'r-')
plt.show()
plt.close()
