import numpy as np
import matplotlib.pyplot as plt


def perlin(out_shape, subdivs, seed=0):
    def f(x):
        return x**3*(x*(x*6-15)+10)
    
    np.random.seed(seed)

    rand_theta = np.random.uniform(0.0,2*np.pi,(subdivs[0]+1)*(subdivs[1]+1)).reshape((subdivs[0]+1),(subdivs[1]+1))
    grad_grid = np.dstack([np.cos(rand_theta), np.sin(rand_theta)])

    delta = (subdivs[0]/out_shape[0], subdivs[1]/out_shape[1])
    d0, d1 = (out_shape[0]//subdivs[0], out_shape[1]//subdivs[1])
    out_grid = np.mgrid[0:subdivs[0]:delta[0], 0:subdivs[1]:delta[1]].transpose(1,2,0)%1
    
    g00 = grad_grid[:-1,:-1].repeat(d0,0).repeat(d1,1)
    g01 = grad_grid[:-1,1:].repeat(d0,0).repeat(d1,1)
    g10 = grad_grid[1:,:-1].repeat(d0,0).repeat(d1,1)
    g11 = grad_grid[1:,1:].repeat(d0,0).repeat(d1,1)
    n00=np.sum((out_grid-np.array([0,0]))*g00, axis=2)
    n01=np.sum((out_grid-np.array([0,1]))*g01, axis=2)
    n10=np.sum((out_grid-np.array([1,0]))*g10, axis=2)
    n11=np.sum((out_grid-np.array([1,1]))*g11, axis=2)

    t = f(out_grid)
    n0 = n00*(1-t[:,:,0]) + n10*t[:,:,0]
    n1 = n01*(1-t[:,:,0]) + n11*t[:,:,0]
    return n0*(1-t[:,:,1]) + n1*t[:,:,1]

    
 

# EDIT : generating noise at multiple frequencies and adding them up
p = np.zeros((128,128))
subdivs=(8,8)
for i in range(4):
    freq = 2**i
    p = perlin(p.shape, (subdivs[0]*freq,subdivs[1]*freq)) / freq+p
# p = my_perlin(p.shape,subdivs)

plt.imshow(p, origin='upper')
plt.show()