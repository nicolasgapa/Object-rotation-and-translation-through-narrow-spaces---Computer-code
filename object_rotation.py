# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:56:12 2020

Nicolas Gachancipa

"""
# Imports.
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import matplotlib.patches as patches

# Inputs.
a = 2                   # Rectangle height (across tube).
b = 3                   # Rectangle width (along tube)
L = 5                   # Tube width.
total_pts_per_x = 100   # Definition (number of points in the mesh across the tube)
x_range = 20            # X and y maximum range (for the plot).

# Simulation (Only works on the lower section of the tube for now, i.e. 
# the obj_y position must be smaller than the tube width L).
obj_x = 4.2   # Initial x.
obj_y = 3.9   # Initial y.

# Define object.
class rectangle:
    def __init__(self, x, y, a, b, alpha=0):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.alpha = alpha

    @property
    def left_bottom(self):
        bottom, left = self.y - self.a/2, self.x - self.b/2
        alpha_rad = self.alpha*mt.pi/180
        l = mt.cos(alpha_rad) * (left - self.x) - mt.sin(alpha_rad) * (bottom - self.y) + self.x
        b = mt.sin(alpha_rad) * (left - self.x) + mt.cos(alpha_rad) * (bottom - self.y) + self.y
        return (l, b)
        
    def get_plot(self):
        return patches.Rectangle(self.left_bottom, self.b, self.a, 
                             edgecolor='r', facecolor='none', angle=self.alpha)
    
# Create x and y arrays.
length = x_range*total_pts_per_x/L
y_array = np.linspace(L/total_pts_per_x, L, total_pts_per_x, endpoint=False)
y_array = L - np.array([y_array,]*(int(length))).transpose()
x_array = np.array([np.linspace(x_range/length, x_range, int(length), endpoint=False),]*total_pts_per_x)

# Compute alphas. 
m = L
c = 0.5*mt.sqrt(a**2 + b**2)
beta = mt.atan(a/b)
heights = (L/2 - abs((L/2) - y_array))/c
alphas = np.where(heights >= 1, mt.pi/2, np.arcsin(heights) - beta)
alphas = np.where(alphas>0, alphas, 0)

# Compute xts.
xts = x_array - c*np.cos(beta + alphas)

# Compute masks.
mask_1 = np.logical_and(xts < m, xts > 0)
mask_2 = np.logical_and(x_array >= m, alphas > 0)
mask_3 = np.logical_and(x_array < m, alphas > 0)

# Case 1: X >= m and Xt < m.
def func_1(x_array, y_array):
    s = x_array - m
    d = np.sqrt(s**2 + (L/2 - abs((L/2) - y_array))**2)
    e = np.sqrt(d**2 - (a/2)**2)
    phi = np.arctan(2*e/a)
    mu = np.arccos((L/2 - abs((L/2) - y_array))/d)
    t = phi - mu
    f = e*np.sin(t)
    g = (L/2 - abs((L/2) - y_array)) - f
    tri = np.arcsin(2*g/a)
    alpha = (np.pi/2) - tri
    return alpha
case_1_mask = np.logical_and(mask_1, mask_2)
case_1 = np.where(case_1_mask, func_1(x_array, y_array), 0)
case_1 = np.where(np.logical_and(case_1 > 0, y_array <= L/2), case_1, 0)

# Case 2: X < m and Xt < m.
def func_2(x_array, y_array):
    s = m - x_array
    d = np.sqrt(s**2 + (L/2 - abs((L/2) - y_array))**2)
    tri = np.arcsin((L/2 - abs((L/2) - y_array))/d)
    phi = np.arccos(a/(2*d))
    subt = np.where(tri >= np.pi/4, tri - phi, tri + phi)
    alpha = (np.pi/2) - subt
    return alpha
case_2_mask = np.logical_and(mask_1, mask_3)
case_2 = np.where(case_2_mask, func_2(x_array, y_array), 0)
case_2 = np.where(np.logical_and(case_2 > 0, y_array <= L/2), case_2, 0)
case_2 = np.where(case_2 >= mt.pi/2, mt.pi/2, case_2)

# Corner.
max_size, min_size = max(alphas.shape), min(alphas.shape)
full_array = np.zeros((max_size, max_size))
full_array[:min_size, :] = alphas 
diagonal = np.flip(np.where(x_array >= y_array, 1, 0), axis=0)
full_array[:min_size, :] *= diagonal

# Plot.
cases = case_1 + case_2
mask_f = (cases > 0)
full_array[:min_size, :][mask_f] = cases[mask_f]
full_array_2 = np.rot90(np.fliplr(full_array))
mask_g = (full_array_2 > 0)
full_array[mask_g] = full_array_2[mask_g]
nodes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cmap = plt.cm.get_cmap('Blues')
full_array[min_size:, min_size:] += -1
fig, ax = plt.subplots(1, dpi=300)
full_array *= (180/mt.pi)
neg = ax.matshow(full_array, origin='lower', extent=[0, x_range, 0, x_range], cmap=cmap)

# Simulate rectangle.
obj = rectangle(obj_x, obj_y, a, b)   
if L/2 - abs((L/2) - obj.y) < (a/2):
    print('The selected initial position is not valid. The rectangle must fit inside the tube.')
else:
    print('\nSimulating:')
    
    # Obtain alpha max.
    x_idx = np.where(x_array[0] == min(x_array[0], key=lambda x:abs(x-obj.x)))[0][0]
    y_idx = np.where(y_array[:, 0] == min(y_array[:, 0], key=lambda y:abs(y-obj.y)))[0][0]
    alpha_max = full_array[total_pts_per_x - y_idx][x_idx] 
    
    # Simulate.
    if alpha_max == 90:
        alpha_max = 360
    print('Alpha max: ', alpha_max, 'degrees.')
    for i in range(0, mt.ceil(alpha_max + 1), 10):
        i = 360 - i
        obj.alpha = i
        fig, ax = plt.subplots(1, dpi=300)
        ax.matshow(full_array, origin='lower', extent=[0, x_range, 0, x_range], cmap=cmap)
        ax.plot(obj.x, obj.y, 'o', color='black', markersize=2)
        ax.add_patch(obj.get_plot())
        cbar = fig.colorbar(neg, ax=ax, orientation='horizontal', boundaries = [i for i in range(91)])
        cbar.set_ticks([0, 45, 90])
        cbar.set_ticklabels([0, 45, 90])
        plt.show()

