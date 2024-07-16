#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:28:28 2024

@author: hma
"""

import matplotlib.pyplot as plt 
import numpy as np
import pickle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def load_data(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def draw_ts_model_based():
    print("111")  
    weight = [0.1, 0.2, 0.3, 0.4, 0.5]
    J = [0.515, 0.382, 0.249, 0.156, 0.152]
    side_payment = [1.342, 1.332, 1.323, 0.115, 0]
    value = [0.649, 0.648, 0.646, 0.202, 0.152]
    att_value = [0.846, 0.837, 0.831, 0.543, 0.531]
    plt.plot(weight, J, marker = "*", markersize = 8, linewidth = 2, color = 'blue', label = "Total payoff J")
    plt.plot(weight, side_payment, marker = "^", markersize = 8, linewidth = 2, color = 'red', label = r'${||\vec{x}||}_1$')
    plt.plot(weight, value, marker = "o", markersize = 8, linewidth = 2, color = 'green', label = r'$V_1(\pi)$')
    plt.plot(weight, att_value, marker = "s", markersize = 8, linewidth = 2, color = 'black', label = r'$V_2(\pi)$')
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    plt.xlabel("c", fontsize = 16)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    plt.ylabel("Value", fontsize = 16) 
    plt.legend()
    # plt.savefig("test.eps")
    plt.show()
    # plt.savefig("test.svg")

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

def draw_ts_model_free_zoomin():
    J_model_base = load_data("../TS_data/Jlist_modelbase.pkl")
    J_model_free20 = load_data("../TS_data/Jlist_modelfree20.pkl")
    J_model_free50 = load_data("../TS_data/Jlist_modelfree50.pkl")
    J_model_free100 = load_data("../TS_data/Jlist_modelfree100.pkl")
    J_model_free200 = load_data("../TS_data/Jlist_modelfree200.pkl")
  
    iterations = np.arange(len(J_model_base))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    ax.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    ax.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    ax.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Iterations", fontsize = 16)
    ax.set_yticks([0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40])
    ax.set_ylabel("J Value", fontsize = 16) 
    ax.legend()
    # axins = inset_axes(ax, width="40%", height="30%", loc='lower right',
                   # bbox_to_anchor=(0.1, 0.1, 1, 1),
                   # bbox_transform=ax.transAxes)

    axins = ax.inset_axes((0.5, 0.1, 0.45, 0.4))
    axins.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    axins.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    axins.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    axins.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    # plt.show()
    zone_left = 150
    zone_right =200
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05
    
    xlim0 = iterations[zone_left]-(iterations[zone_right]-iterations[zone_left])*x_ratio
    xlim1 = iterations[zone_right]+(iterations[zone_right]-iterations[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((J_model_base[zone_left:zone_right], J_model_free20[zone_left:zone_right],  
                   J_model_free100[zone_left:zone_right],  J_model_free200[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")

    xy = (xlim0,ylim0)
    xy2 = (xlim0,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                          axesA=axins,axesB=ax)
    axins.add_artist(con)

    xy = (xlim1,ylim0)
    xy2 = (xlim1,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                          axesA=axins,axesB=ax)
    
    axins.add_artist(con)

    plt.savefig("test.pdf",format="pdf")
    plt.show()


    
def draw_gw_model_based():
    weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    J = [2.82, 2.529, 2.24, 1.953, 1.667, 1.382, 1.098, 0.814, 0.531, 0.248, 0.001]
    side_payment = [3.032, 2.899, 2.87, 2.87, 2.855, 2.846, 2.84, 2.839, 2.829, 2.825, 0]
    value = [3.123, 3.109, 3.101, 3.101, 3.095, 3.089, 3.086, 3.085, 3.077, 3.073, 0.001]
    att_value = [9.51, 9.145, 9.051, 9.017, 8.969, 8.918, 8.908, 8.886, 8.879, 8.863, 8.242]
    
    plt.plot(weight, J, marker = "*", markersize = 8, linewidth = 2, color = 'blue', label = "Total payoff J")
    plt.plot(weight, side_payment, marker = "^", markersize = 8, linewidth = 2, color = 'red', label = r'${||\vec{x}||}_1$')
    plt.plot(weight, value, marker = "o", markersize = 8, linewidth = 2, color = 'green', label = r'$V_1(\pi)$')
    plt.plot(weight, att_value, marker = "s", markersize = 8, linewidth = 2, color = 'black', label = r'$V_2(\pi)$')
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.xlabel("c", fontsize = 16)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.ylabel("Value", fontsize = 16) 
    plt.legend(loc = (0.02, 0.5))
    plt.savefig('GW_model_based.eps')
    plt.show()
    
def draw_gw_model_based_2():
    weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    J = [5.43, 4.603, 3.788, 2.981, 2.181, 1.386, 0.595, 0.001]
    side_payment = [8.295, 8.194, 8.1, 8.019, 7.974, 7.926, 7.892, 0]
    value = [6.26, 6.241, 6.218, 6.189, 6.168, 6.141, 6.119, 0.001]
    att_value = [9.029, 8.967, 8.908, 8.855, 8.828, 8.8, 8.777, 8.242]
    
    plt.plot(weight, J, marker = "*", markersize = 8, linewidth = 2, color = 'blue', label = "Total payoff J")
    plt.plot(weight, side_payment, marker = "^", markersize = 8, linewidth = 2, color = 'red', label = r'${||\vec{x}||}_1$')
    plt.plot(weight, value, marker = "o", markersize = 8, linewidth = 2, color = 'green', label = r'$V_1(\pi)$')
    plt.plot(weight, att_value, marker = "s", markersize = 8, linewidth = 2, color = 'black', label = r'$V_1(\pi)$')
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    plt.xlabel("c", fontsize = 16)
    plt.yticks([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    plt.ylabel("Value", fontsize = 16) 
    plt.legend()
    plt.savefig('GW2_model_based.eps')
    plt.show()
    

def draw_gw_model_free():
    
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
    plt.yticks([1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25])
    plt.ylabel("J Value", fontsize = 16) 
    plt.legend()
    plt.show()
    

def draw_GW_model_free_zoomin():
    J_model_base = load_data("../GW_data/Jlist_modelbase.pkl")
    J_model_free20 = load_data("../GW_data/Jlist_modelfree20.pkl")
    J_model_free50 = load_data("../GW_data/Jlist_modelfree50.pkl")
    J_model_free100 = load_data("../GW_data/Jlist_modelfree100.pkl")
    J_model_free200 = load_data("../GW_data/Jlist_modelfree200.pkl")
    
    iterations = np.arange(len(J_model_base))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    ax.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    ax.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    ax.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Iterations", fontsize = 16)
    ax.set_yticks([1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25])
    ax.set_ylabel("J Value", fontsize = 16) 
    ax.legend()
    
    # axins = inset_axes(ax, width="40%", height="30%", loc='lower right',
                   # bbox_to_anchor=(0.1, 0.1, 1, 1),
                   # bbox_transform=ax.transAxes)
    
    axins = ax.inset_axes((0.5, 0.1, 0.45, 0.4))
    axins.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    axins.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    axins.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    axins.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    # plt.show()
    
    zone_left = 140
    zone_right =200
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05
    
    xlim0 = iterations[zone_left]-(iterations[zone_right]-iterations[zone_left])*x_ratio
    xlim1 = iterations[zone_right]+(iterations[zone_right]-iterations[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((J_model_base[zone_left:zone_right], J_model_free20[zone_left:zone_right],  
                   J_model_free100[zone_left:zone_right],  J_model_free200[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")
    
    xy = (xlim0,ylim0)
    xy2 = (xlim0,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                          axesA=axins,axesB=ax)
    axins.add_artist(con)

    xy = (xlim1,ylim0)
    xy2 = (xlim1,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                          axesA=axins,axesB=ax)
    axins.add_artist(con)
    plt.savefig('GW_model_free.eps')


def draw_GW_model_free_zoomin_2():
    J_model_base = load_data("../GW2_data/Jlist_modelbase.pkl")
    J_model_free20 = load_data("../GW2_data/Jlist_modelfree20.pkl")
    J_model_free50 = load_data("../GW2_data/Jlist_modelfree50.pkl")
    J_model_free100 = load_data("../GW2_data/Jlist_modelfree100.pkl")
    J_model_free200 = load_data("../GW2_data/Jlist_modelfree200.pkl")
    
    iterations = np.arange(len(J_model_base))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    ax.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    ax.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    ax.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Iterations", fontsize = 16)
    ax.set_yticks([3.64, 3.68, 3.72, 3.76, 3.8])
    ax.set_ylabel("J Value", fontsize = 16) 
    ax.legend()
    
    # axins = inset_axes(ax, width="40%", height="30%", loc='lower right',
                   # bbox_to_anchor=(0.1, 0.1, 1, 1),
                   # bbox_transform=ax.transAxes)
    
    axins = ax.inset_axes((0.5, 0.1, 0.45, 0.4))
    axins.plot(iterations, J_model_base, linewidth = 1.5, color = 'blue', label = "Model based")
    axins.plot(iterations, J_model_free20, linewidth = 1.5, color = 'red', label = "Sample size: 20")
    # plt.plot(iterations, J_model_free50, marker = ".", markersize = 8, linewidth = 1, color = 'green', label = "Sample size: 50")
    axins.plot(iterations, J_model_free100, linewidth = 1.5, color = 'black', label = "Sample size: 100")
    axins.plot(iterations, J_model_free200, linewidth = 1.5, color = 'cyan', label = "Sample size: 200")
    # plt.show()
    
    zone_left = 140
    zone_right =200
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05
    
    xlim0 = iterations[zone_left]-(iterations[zone_right]-iterations[zone_left])*x_ratio
    xlim1 = iterations[zone_right]+(iterations[zone_right]-iterations[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((J_model_base[zone_left:zone_right], J_model_free20[zone_left:zone_right],  
                   J_model_free100[zone_left:zone_right],  J_model_free200[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")
    
    xy = (xlim0,ylim0)
    xy2 = (xlim0,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                          axesA=axins,axesB=ax)
    axins.add_artist(con)

    xy = (xlim1,ylim0)
    xy2 = (xlim1,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                          axesA=axins,axesB=ax)
    axins.add_artist(con)
    plt.savefig('GW2_model_free.eps')
if __name__ == "__main__":
    # draw_ts_model_based()
    # draw_ts_model_free_zoomin()
    # draw_gw_model_based()
    # draw_gw_model_free()
    # draw_gw_model_based_2()
    draw_GW_model_free_zoomin_2()