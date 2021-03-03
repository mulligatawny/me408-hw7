###############################################################################
# 1D Advection Equation Solver using Tau's Method with Chebyshev Expansions   #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

N = 16
t = np.arange(0, N+1)*np.pi/N # uniform grid
#x = np.flip(np.cos(t))        # Chebyshev grid
x = np.cos(t)

# initial condition
u0 = np.zeros_like(x, dtype='float')
u0[x<-0.5] = np.sin(2*np.pi*x[x<-0.5]) + 1/3*np.sin(6*np.pi*x[x<-0.5])
# Chebyshev transform
y0 = cheby.cheby(u0)
k = np.arange(0, N+1)

def fun(t, y):
    s = 0.0
    for p in range(i+1,N+1):
        if (p+i)%2 == 1: 
            if i == 0 or N:
                s = s + p*y[p]
            else:
                s = s + 2*p*y[p]
    return -s

dt = 0.0001
t = 0.0
y = y0
yn = np.zeros_like(x, dtype='complex')

# Runge-Kutta IV time integrator
while t < 0.2:
    for i in range(N):
        k1 = dt*fun(t, y)
        k2 = dt*fun(t+dt/2, y+k1/2)
        k3 = dt*fun(t+dt/2, y+k2/2)
        k4 = dt*fun(t+dt, y+k3)
        yn[i] = y[i] + k1/6 + k2/3 + k3/3 + k4/6
    yn[N] = -sum(yn[:-1:2]) + sum(yn[1::2]) # BC
    y = yn
    t = t + dt

ue = np.roll(u0, -1)

ye = cheby.icheby(yn)
plt.plot(x, np.real(ye), color='salmon', label='numerical')
plt.plot(x, ue, 'k-.', label='analytical')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('Solution without BC at t = {:.2f}'.format(t))
plt.show()
