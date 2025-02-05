import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
N = 8

g = 0.75
tau = 5.0
epsilon = 0.1
amp = 2.0

L = 2 * np.pi
dx = L / N
x = (np.arange(N) * dx - np.pi).reshape(-1, 1)

# Construct W matrix
Wrow = np.cos(np.arange(N) * 2 * np.pi / N)
Wmat = np.zeros((N, N))
for nrow in range(N):
    Wmat[nrow, :] = np.roll(Wrow, nrow)

# Time parameters
tfinal = 2000
t = 0
dt = 0.01
sig = 0.25

theta0 = 0
v00 = amp * (1.0 + epsilon * (np.cos(x - theta0) - 1))
v = v00.copy()

T = [t]
th = theta0
thsave = [th]

xmean = np.mean(v * np.cos(x))
ymean = np.mean(v * np.sin(x))

thout = [np.arctan2(ymean, xmean)]

# Function for ringrhs
def ringrhs(t, v, Wmat, tau, g, N, v00):
    nonlin = (2 * g / N) * np.dot(Wmat, v) + v00
    nonlin[nonlin < 0] = 0
    dydt = (-v + nonlin) / tau
    return dydt

# Main loop
k = 0
while t < tfinal:
    v00 = amp * (1.0 + epsilon * (np.cos(x - th) - 1))
    v = v + dt * ringrhs(t, v, Wmat, tau, g, N, v00)
    th = th + sig * np.sqrt(dt) * np.random.randn()
    t = t + dt
    k = k + 1

    T.append(t)
    thsave.append(th)

    xmean = np.mean(v * np.cos(x))
    ymean = np.mean(v * np.sin(x))

    thout.append(np.arctan2(ymean, xmean))

    if k % 100 == 0:
        plt.figure(1)
        plt.plot(np.append(x, L/2), np.append(v, v[0]), '.', np.append(x, L/2), np.append(v00, v00[0]), '-')
        plt.axis([-L/2, L/2, 0, 1.1 * np.max(v)])
        plt.xlabel('x/π')
        plt.ylabel('v')
        plt.draw()
        plt.pause(0.01)

thsave = np.unwrap(thsave)
thout = np.unwrap(thout)

plt.figure(2)
plt.plot(T, thsave, T, thout)
plt.show()