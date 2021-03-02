import numpy as np
import matplotlib.pyplot as plt

N = 64
x = np.linspace(0, 2*np.pi, N+1)[:-1]
y = np.sin(4*x) + 0.15*np.sin(14*x)

# one-sided finite difference to compute derivative
def compute_der(y):
    dx = 2*np.pi/N
    yp = np.zeros_like(y)
    yp[1:] = (y[1:] - y[0:-1])/dx
    yp[0] = (y[0]-y[-1])/dx
    return yp
# FFT 
def compute_FFT(y):
    return (np.fft.fft(y))/N

def compute_IFFT(Fk):
    return np.fft.ifft(Fk)*N

def unaliased(y):
    LHS = y*compute_der(y)
    RHS = compute_der((y**2)/2)
    Fk_LHS = compute_FFT(LHS)
    Fk_RHS = compute_FFT(RHS)
    k = np.arange(-N/2, N/2)
    k = np.fft.fftshift(k)

#    plt.plot(k, np.abs(Fk_LHS), 'ro-', label='LHS')
#    plt.plot(k, np.abs(Fk_RHS), 'ko-', label='RHS')
#    plt.plot(x, np.real(compute_IFFT(Fk_LHS)), label='LHS-aliased')
#    plt.plot(x, np.real(compute_IFFT(Fk_RHS)))
    plt.xlabel('$k$')
    plt.ylabel('Fourier coefficients')
    plt.legend()
unaliased(y)

u_L = y
# Step 1: perform FFT on u
Fu_L = compute_FFT(u_L)
# Step 2: pad coefficients with 0s
Fu_L = np.pad(Fu_L, int(N/4), 'constant') # pad with zeros
# Step 3: inverse transform to get u'
u_Lr = compute_IFFT(Fu_L)

# Same for v
v_L = compute_der(y)
Fv_L = compute_FFT(v_L)
Fv_L = np.pad(Fv_L, int(N/4), 'constant')
v_Lr = compute_IFFT(Fv_L)

# Step 4: perform FFT on u'v'
upvp_L = compute_FFT(u_Lr*v_Lr)
# Step 5: truncate coefficients
tr_L = upvp_L[int(N/4):int(N+N/4)]
# Step 6: inverse transform to get uv
LHS = compute_IFFT(tr_L)
#plt.plot(x, np.real(LHS), 'ro-', label='LHS-unaliased')
plt.show()
