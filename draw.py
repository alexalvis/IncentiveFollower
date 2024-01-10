#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:28:28 2024

@author: hma
"""

import matplotlib.pyplot as plt 
import numpy as np
import pickle

def load_data(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def draw_ts_model_based():
    weight = [0.1, 0.2, 0.3, 0.4]
    J = [0.515, 0.382, 0.249, 0.155]
    side_payment = [1.342, 1.331, 1.322, 0.108]
    value = [0.649, 0.648, 0.646, 0.198]
    
    plt.plot(weight, J, marker = "*", markersize = 8, linewidth = 2, color = 'blue', label = "J value")
    plt.plot(weight, side_payment, marker = "^", markersize = 8, linewidth = 2, color = 'red', label = "Side-payment")
    plt.plot(weight, value, marker = "o", markersize = 8, linewidth = 2, color = 'green', label = "Leader's value")
#    plt.plot(x, y4, marker = "s", markersize = 8, linewidth = 2, color = 'black', label = "Initial policy 4")
    plt.xticks([0.1, 0.2, 0.3, 0.4])
    plt.xlabel("Weight", fontsize = 16)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    plt.ylabel("Value", fontsize = 16) 
    plt.legend()
    plt.show()
    

def draw_ts_model_free():
    
    J_model_base = load_data("Jlist_modelbase.pkl")
    J_model_free20 = load_data("Jlist_modelfree20.pkl")
    J_model_free50 = load_data("Jlist_modelfree50.pkl")
    J_model_free100 = load_data("Jlist_modelfree100.pkl")
    J_model_free200 = load_data("Jlist_modelfree200.pkl")
    
    iterations = np.arange(len(J_model_base))
    
    plt.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    plt.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    plt.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    plt.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    plt.xticks([50, 100, 150, 200])
    plt.xlabel("Iterations", fontsize = 16)
    plt.yticks([0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40])
    plt.ylabel("J Value", fontsize = 16) 
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    # draw_ts_model_based()
    draw_ts_model_free()