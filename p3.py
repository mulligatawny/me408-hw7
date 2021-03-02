import numpy as np
import matplotlib.pyplot as plt

N = 32
x = np.linspace(0, 2*np.pi, N+1)[:-1]
y1 = np.sin(4*x) + 0.15*np.sin(14*x) 
y2 = np.sin(8*x) + 0.2*np.cos(14*x)
f = y1*y2

Fk = np.fft.fft(f)/N
Fk = np.fft.fftshift(Fk)
k = np.arange(-N/2, N/2)
#k = np.fft.fftshift(k)

# product defined analytically
N2 = 256
x = np.linspace(0, 2*np.pi, N2+1)[:-1]
y3 = 0.5*(-np.cos(12*x) + np.cos(4*x)) +\
     (0.2/2)*(np.sin(18*x) + np.sin(10*x)) +\
     (0.15/2)*(-np.cos(22*x) + np.cos(6*x)) +\
     (0.03/2)*(np.sin(28*x))
Fkt = np.fft.fft(y3)/N2
Fkt = np.fft.fftshift(Fkt)
k2 = np.arange(-N2/2, N2/2)

print(np.abs(Fk))
plt.plot(k, np.abs(Fk), '-o',color='teal')
plt.plot(k2, np.abs(Fkt), '-',color='orangered')
plt.ylabel('Fourier coefficients')
plt.xlabel('$k$')
plt.xlim([-17,17])
plt.grid('on')
plt.show()
