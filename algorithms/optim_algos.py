# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:36:05 2017

@author: Yining Cai
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1000)
n = 10000
x0 = np.ones(n)
x1 = np.random.normal(loc = 20, scale = 3, size = n)
x2 = np.random.normal(loc = 15, scale = 2, size = n)
x3 = np.random.normal(loc = 40, scale = 4, size = n)
t = x0*0 + x1*3 - x2*5 + x3 + np.random.normal(loc = 0, scale = 5, size = n)

x = np.stack((x0, x1, x2, x3))
m, n = x.shape
w = np.random.uniform(size = m)
y = np.matmul(w, x) 

def loss(x, w, t):
    y = np.matmul(w, x)
    return(np.sum(np.square(t-y))/n)

def gradientDescent(x, w, t, learning_rate):
    max_iter = 10000
    learning = np.zeros(max_iter//10)
    for epoch in range(max_iter):
        if epoch % 10 == 0:
            print(epoch)
            learning[epoch//10] = loss(x, w, t)
        dw = -np.matmul(x, t - np.matmul(w, x))/n
        new_w = w - learning_rate * dw
        if np.sum(abs(new_w - w)) < 1e-5:
            w = new_w
            break
        else:
            w = new_w
            
    learning = learning[:epoch//10]
    return(w, learning)

def GDMomentum(x, w, t, learning_rate):
    max_iter = 10000
    learning = np.zeros(max_iter//10)
    v = 0
    momentum = 0.8
    for epoch in range(max_iter):
        if epoch % 10 == 0:
            print(epoch)
            learning[epoch//10] = loss(x, w, t)
        
        dw = -np.matmul(x, t - np.matmul(w, x))/n
        v = momentum*v + learning_rate*dw
        new_w = w - v
        if np.sum(abs(new_w - w)) < 1e-5:
            w = new_w
            break
        else:
            w = new_w
            
    learning = learning[:epoch//10]
    return(w, learning)


def nesterovMomentum(x, w, t, learning_rate):
    max_iter = 10000
    learning = np.zeros(max_iter//10)
    v = 0
    momentum = 0.8
    for epoch in range(max_iter):
        if epoch % 10 == 0:
            print(epoch)
            learning[epoch//10] = loss(x, w, t)
        
        dw = -np.matmul(x, t - np.matmul(w, x))/n
        v_new = momentum *v + learning_rate*dw
        v_nesterov = (1+momentum)*v_new + momentum*v
        new_w = w - v_nesterov
        v = v_new
        if np.sum(abs(new_w - w)) < 1e-5:
            w = new_w
            break
        else:
            w = new_w
            
    learning = learning[:epoch//10]
    return(w, learning)


w_gd, learning_gd = gradientDescent(x, w, t, 5e-5)
w_gdm, learning_gdm = GDMomentum(x, w, t, 5e-5)
w_nm, learning_nm = nesterovMomentum(x, w, t, 5e-5)

plt.plot(learning_gd, color = 'blue')
plt.plot(learning_gdm, color = 'green')
plt.plot(learning_nm, color = 'red')