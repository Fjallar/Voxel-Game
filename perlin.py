import numpy as np
import matplotlib.pyplot as plt
import rgen
from ctypes import c_uint32

seed=7937362



#pos_xy is subdivided
def perlin(out_shape, pos_xy, octave=0):
    def f(x):
        return x**3*(x*(x*6-15)+10)
    # np.random.seed(seed)¨
    rgrid = rgen.from_pos(pos_xy.astype("uint32"), octave)/2**32
    rand_theta = 2*np.pi*(rgrid)  
    sd_x, sd_y = rand_theta.shape
    
    grad_grid = np.dstack([np.cos(rand_theta), np.sin(rand_theta)])
    sd_x, sd_y = rand_theta.shape
    sd_x-=1;sd_y-=1

    delta = (sd_x/out_shape[0], sd_y/out_shape[1])
    d0, d1 = (out_shape[0]//sd_x, out_shape[1]//sd_y)
    out_grid = np.mgrid[0:sd_x:delta[0], 0:sd_y:delta[1]].transpose(1,2,0)%1
    
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

#Questions:
#1. Will x=[0,128],y=[0,128] overlap with x= [16,112], y=[-64,64] @ x=[16,112] y=[0,64]?
#2. can I do a 3D version of this?
#3. How does opensimplex noise work?
#4. 

#max_stepsize
def perlin_fractal(pos_xy, max_stepsize=(16,16), octaves=4):
    # EDIT : generating noise at multiple frequencies and adding them up
    p = np.zeros_like(pos_xy[0,:-1,:-1],dtype=np.float64)
    for octave in range(octaves):
        freq = 2**octave
        step_x, step_y = (int(max_stepsize[0]>>octave), int(max_stepsize[1]>>octave))
        p += perlin(p.shape, pos_xy[:,::step_x,::step_y], octave=octave)/freq
    return p*np.sqrt(2)

if __name__ == "__main__":
    pos_xy=np.stack(np.meshgrid(np.arange(129),np.arange(129)))
    p = perlin_fractal(pos_xy)
    plt.imshow(p, origin='upper')
    plt.show()