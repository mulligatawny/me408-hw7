import numpy as np
import matplotlib.pyplot as plt

N = 32
x = np.linspace(0, 2*np.pi, N+1)[:-1]
#y = np.sin(4*x) + 0.15*np.sin(14*x)
y = np.cos(x)
yp = 4*np.sin(4*x) + 0.21*np.cos(14*x)
k = np.arange(-N/2, N/2)
#k = np.fft.fftshift(k)

def compute_FFT(y):
    return np.fft.fftshift(np.fft.fft(y))/(len(y))

def compute_IFFT(Fk):
    return np.fft.fftshift(np.fft.ifft(Fk))*(len(Fk))

Fk = compute_FFT(y)
fr = compute_IFFT(Fk)
plt.plot(x, np.real(fr))

#u = y
#Fu = compute_FFT(u)
#Fu = np.pad(Fu, int(N/4), 'constant')
#
#ur = compute_IFFT(Fu)
#
#v = yp
#Fv = compute_FFT(v)
#Fv = np.pad(Fv, int(N/4), 'constant')
#vr = compute_IFFT(Fv)
#
#uv = ur*vr
#Fuv = compute_FFT(uv)
#Tuv = Fuv[int(N/4):int(N+N/4)]
#uvr = compute_IFFT(Tuv)
#
#plt.plot(x, np.real(uvr))
#plt.plot(x, y*yp, 'r:')
plt.show()
