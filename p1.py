import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

start = time.time()

N = 64
nu = 0.4
x = np.linspace(0, 2*np.pi, N+1)[:-1]
u0 = 5*np.sin(x) # initial condition
y0 = np.fft.fft(u0)/N
k = np.arange(-N/2, N/2)
k = np.fft.fftshift(k)

def fun(t, y):
    df = -nu*(k[i]**2)*y[i]
    ad = 0.0
    for m in range(N):
        for n in range(N):
            if k[m]+k[n] == k[i]:
                ad = ad + 1j*k[m]*y[m]*y[n]
    return -ad + df

dt = 0.004
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
    y = yn
    t = t + dt

end = time.time()
ye = np.fft.ifft(y)*N
plt.plot(x, u0, 'k-', label='I.C.')
plt.plot(x, np.real(ye), 'o-', color='orangered', label='t = {:.1f}'.format(t))
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.show()
print(end-start)
