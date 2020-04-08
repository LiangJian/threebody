import numpy as np
import numba
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

# TODO
# 1, catch impact event in the solver
# 2， 3-D case
# 3, collect stable initial conditions

@numba.jit
def threebody2d(t, y, m0, m1, m2):

    r0_x, v0_x, r0_y, v0_y, r1_x, v1_x, r1_y, v1_y, r2_x, v2_x, r2_y, v2_y = y

    d0 = ((r2_x-r1_x)**2 + (r2_y-r1_y)**2)**(3/2.)  # 1-2 distance
    d1 = ((r0_x-r2_x)**2 + (r0_y-r2_y)**2)**(3/2.)  # 0-2 distance
    d2 = ((r1_x-r0_x)**2 + (r1_y-r0_y)**2)**(3/2.)  # 0-1 distance

    if d0 < 1e-6 or d1 < 1e-6 or d2 < 1e-6:
        return [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]

    dydt = []
    dydt.append(v0_x)
    dydt.append(-m1 * (r0_x - r1_x) / d2 - m2 * (r0_x - r2_x) / d1)
    dydt.append(v0_y)
    dydt.append(-m1 * (r0_y - r1_y) / d2 - m2 * (r0_y - r2_y) / d1)

    dydt.append(v1_x)
    dydt.append(-m2 * (r1_x - r2_x) / d0 - m0 * (r1_x - r0_x) / d2)
    dydt.append(v1_y)
    dydt.append(-m2 * (r1_y - r2_y) / d0 - m0 * (r1_y - r0_y) / d2)

    dydt.append(v2_x)
    dydt.append(-m1 * (r2_x - r1_x) / d0 - m0 * (r2_x - r0_x) / d1)
    dydt.append(v2_y)
    dydt.append(-m1 * (r2_y - r1_y) / d0 - m0 * (r2_y - r0_y) / d1)

    return dydt


y0 = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 3.0, 0.0, -2.0, 0.0, -1.0, 0.0]
rx = 0.9700436
ry = -0.24308753
vx = 0.466203685
vy = 0.43236573
y0 = [rx, vx, ry, vy, -rx, vx, -ry, vy, 0.0, -2*vx, 0.0, -2*vy]
sol = solve_ivp(threebody2d, t_span=(0, 63), y0=y0, args=(1.0,1.0,1.0), rtol=1e-10, atol=1e-10)

if sol.status != 0:
    print(sol.message)
    exit(1)

r0_x = sol.y[0, :] 
r0_y = sol.y[2, :] 
r1_x = sol.y[4, :] 
r1_y = sol.y[6, :] 
r2_x = sol.y[8, :] 
r2_y = sol.y[10, :]

N = 500
assert N <= sol.y.shape[1]

plt.plot(r0_x[:N], r0_y[:N])
plt.plot(r1_x[:N], r1_y[:N])
plt.plot(r2_x[:N], r2_y[:N])
plt.show()

fig = plt.figure()
ax1 = plt.axes(xlim=(-1.1, 1.1), ylim=(-0.5, 0.5))
lines = []
for index in range(3):
    lobj = ax1.plot([],[],alpha=1.0,color='C%d'%index)[0]
    lines.append(lobj)
for index in range(3):
    lobj = ax1.plot([],[],markersize=8,marker='o',color='C%d'%index,alpha=1)[0]
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines


def animate(i):

    lines[0].set_data(r0_x[:i+1], r0_y[:i+1])
    lines[1].set_data(r1_x[:i+1], r1_y[:i+1])
    lines[2].set_data(r2_x[:i+1], r2_y[:i+1])
    lines[3].set_data(r0_x[i], r0_y[i])
    lines[4].set_data(r1_x[i], r1_y[i])
    lines[5].set_data(r2_x[i], r2_y[i])

    return lines


anim = animation.FuncAnimation(fig, animate, init_func=init,frames=N, interval=5/N*1000, blit=True, repeat=False)
plt.show()

fig = plt.figure()
ax1 = plt.axes(xlim=(-1.1, 1.1), ylim=(-0.5, 0.5))
N2 = 40
assert N2 < N
lines = []
for index in range(3):
    lobj = ax1.plot([],[],alpha=1.0,color='C%d'%index)[0]
    lines.append(lobj)
for index in range(3):
    lobj = ax1.plot([],[],markersize=8,marker='o',color='C%d'%index,alpha=1)[0]
    lines.append(lobj)


def animate(i):

    if i >= N2:
        lines[0].set_data(r0_x[i-N2:i+1], r0_y[i-N2:i+1])
        lines[1].set_data(r1_x[i-N2:i+1], r1_y[i-N2:i+1])
        lines[2].set_data(r2_x[i-N2:i+1], r2_y[i-N2:i+1])
    else:
        lines[0].set_data(r0_x[:i+1], r0_y[:i+1])
        lines[1].set_data(r1_x[:i+1], r1_y[:i+1])
        lines[2].set_data(r2_x[:i+1], r2_y[:i+1])
    lines[3].set_data(r0_x[i], r0_y[i])
    lines[4].set_data(r1_x[i], r1_y[i])
    lines[5].set_data(r2_x[i], r2_y[i])

    return lines


anim = animation.FuncAnimation(fig, animate, init_func=init,frames=N, interval=5/N*1000, blit=True, repeat=False)
plt.show()
