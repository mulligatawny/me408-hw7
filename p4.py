# 4/4 De-aliasing
import numpy as np
import matplotlib.pyplot as plt

N = 32
x = np.linspace(0, 2*np.pi, N+1)[:-1]
y = np.sin(4*x) + 0.15*np.sin(14*x)
yp = 4*np.cos(4*x) + 2.1*np.cos(14*x)
k = np.arange(-N/2, N/2)

# LHS = y*(dy/dx)
Fk = np.fft.fftshift(np.fft.fft(y))/N
dydx = np.fft.ifft(np.fft.ifftshift(1j*k*Fk))*N
LHS = y*dydx
# RHS = d((y**2)/2)/dx
Fk2 = np.fft.fftshift(np.fft.fft(0.5*y**2))/N
RHS = np.fft.ifft(np.fft.ifftshift(1j*k*Fk2))*N

fig1 = plt.figure(1)
plt.plot(x, np.real(LHS), 'r.-', label='LHS')
plt.plot(x, np.real(RHS), 'g--', label='RHS')
plt.plot(x, y*yp, 'ko', label='exact')
plt.xlabel('$x$')
plt.ylabel('LHS, RHS')
plt.title('Without de-aliasing')
plt.grid()
plt.legend()

# de-aliasing the LHS
def compute_FFT(f):
    return np.fft.fftshift(np.fft.fft(f))/len(f)

def compute_IFFT(F):
    return np.fft.ifft(np.fft.ifftshift(F))*len(F)

u = y
Fu = compute_FFT(u)
Fu = np.pad(Fu, int(N/4), 'constant')
ur = compute_IFFT(Fu)

v = y/2
Fv = compute_FFT(v)
Fv = np.pad(Fv, int(N/4), 'constant')
vr = compute_IFFT(Fv)

uv = ur*vr
Fuv = compute_FFT(uv)
Tuv = Fuv[int(N/4):int(N+N/4)]
uvr = compute_IFFT(Tuv)

Fuvr = compute_FFT(uvr)
RHSd = compute_IFFT(1j*k*Fuvr)

fig2 = plt.figure(2)
plt.plot(x, np.real(LHS), 'r.-', label='LHS')
plt.plot(x, np.real(RHSd), 'g.--', label='RHS')
plt.plot(x, y*yp, 'ko', label='exact')
plt.xlabel('$x$')
plt.ylabel('LHS, RHS')
plt.title('With de-aliasing')
plt.grid()
plt.legend()
plt.show()
