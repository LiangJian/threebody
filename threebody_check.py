import numpy as np
import numba
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

# TODO
# 1, catch impact event in the solver
# 2，3-D case


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


rx = 0.9700436
ry = -0.24308753
vx = 0.466203685
vy = 0.43236573

y0 = [rx, vx, ry, vy, -rx, vx, -ry, vy, 0.0, -2*vx, 0.0, -2*vy]
sol = solve_ivp(threebody2d, t_span=(0, 60), y0=y0, args=(1.0,1.0,1.0), rtol=1e-10, atol=1e-10)

if sol.status != 0:
    print(sol.message)
    exit(1)

r0_x = sol.y[0, :] 
r0_y = sol.y[2, :] 
r1_x = sol.y[4, :] 
r1_y = sol.y[6, :] 
r2_x = sol.y[8, :] 
r2_y = sol.y[10, :]


def period_finding(sol_):
    for it in range(10, sol_.t.size):
        judge = True
        for iv in range(sol_.y.shape[0]):
            judge = judge and abs(sol_.y[iv, it] - sol_.y[iv, 0])**2**0.5 < 1e-3
        if judge:
            return it
    return 0


period = period_finding(sol)
assert period > 0
print('period found:', period, 'steps, T=', sol.t[period])

N = 40
t_max = sol.t[period + 1]
t_play = 5  # s
di = 1  # the delta i

plt.plot(r0_x[:period+1], r0_y[:period+1])
plt.plot(r1_x[:period+1], r1_y[:period+1])
plt.plot(r2_x[:period+1], r2_y[:period+1])
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


def animate1(i):

    lines[0].set_data(r0_x[sol.t<=i], r0_y[sol.t<=i])
    lines[1].set_data(r1_x[sol.t<=i], r1_y[sol.t<=i])
    lines[2].set_data(r2_x[sol.t<=i], r2_y[sol.t<=i])
    lines[3].set_data(r0_x[sol.t<=i][-1], r0_y[sol.t<=i][-1])
    lines[4].set_data(r1_x[sol.t<=i][-1], r1_y[sol.t<=i][-1])
    lines[5].set_data(r2_x[sol.t<=i][-1], r2_y[sol.t<=i][-1])

    return lines


anim = animation.FuncAnimation(fig, animate1, init_func=init,frames=np.linspace(0,t_max,N), interval=t_play*1000/N, blit=True, repeat=True)
anim.save('float-8_1.gif', writer='imagemagick',fps=N//3,dpi=80)
# plt.show()

fig = plt.figure()
ax1 = plt.axes(xlim=(-1.1, 1.1), ylim=(-0.5, 0.5))
lines = []
for index in range(3):
    lobj = ax1.plot([],[],alpha=1.0,color='C%d'%index)[0]
    lines.append(lobj)
for index in range(3):
    lobj = ax1.plot([],[],markersize=8,marker='o',color='C%d'%index,alpha=1)[0]
    lines.append(lobj)

def animate2(i):
    lines[0].set_data(r0_x[np.logical_and(i-di<sol.t,sol.t<=i)], r0_y[np.logical_and(i-di<sol.t,sol.t<=i)])
    lines[1].set_data(r1_x[np.logical_and(i-di<sol.t,sol.t<=i)], r1_y[np.logical_and(i-di<sol.t,sol.t<=i)])
    lines[2].set_data(r2_x[np.logical_and(i-di<sol.t,sol.t<=i)], r2_y[np.logical_and(i-di<sol.t,sol.t<=i)])
    lines[3].set_data(r0_x[sol.t<=i][-1], r0_y[sol.t<=i][-1])
    lines[4].set_data(r1_x[sol.t<=i][-1], r1_y[sol.t<=i][-1])
    lines[5].set_data(r2_x[sol.t<=i][-1], r2_y[sol.t<=i][-1])

    return lines


anim = animation.FuncAnimation(fig, animate2, init_func=init,frames=np.linspace(0,t_max,N), interval=t_play*1000/N, blit=True, repeat=True)
anim.save('float-8_2.gif', writer='imagemagick',fps=N//3,dpi=80)
# plt.show()
