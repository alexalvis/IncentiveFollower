# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:47:47 2023

@author: hma2
"""

import numpy as np
import MDP
import math
import Sample

class GC:
    def __init__(self, mdp, lr_x, policy, epsilon, modify_list, weight):
        self.mdp = mdp
        self.st_len = len(mdp.states)
        self.act_len = len(mdp.actions)
        self.x_size = self.st_len * self.act_len
        self.base_reward = reward2list(mdp.reward, mdp.states, mdp.actions)
        self.x = np.zeros(self.x_size)
        self.lr_x = lr_x
        self.tau = self.mdp.tau
        self.policy = policy
        self.policy_m = self.convert_policy()
        self.P_matrix = self.construct_P()
        self.epsilon = epsilon
        self.modify_list = modify_list
        self.weight = weight
        self.sample = Sample.SampleTraj(self.mdp)
        
    def reward_sidepay(self):
        return self.base_reward + self.x
    
    def J_func(self):
        reward_s = self.reward_sidepay()
        V, policy = self.mdp.get_policy_entropy(reward_s)
        J = self.mdp.init.dot(V) + self.h_func()
        return J, policy

    def h_func(self):
        #Define the h function
        return -self.weight * np.sum(self.x)
    
    def construct_P(self):
        P = np.zeros((self.x_size, self.x_size))
        for i in range(self.st_len):
            for j in range(self.act_len):
                for next_st, pro in self.mdp.transition[self.states[i]][self.actions[j]].items():
                    next_index = self.mdp.states.index(next_st)
                    P[i * self.act_len + j][next_index * self.act_len : (next_index + 1) * self.act_len] = pro
        return P
    
    def convert_policy(self):
        policy_m = np.zeros(self.x_size)
        i = 0
        for st in self.mdp.states:
            for act in self.mdp.actions:
                policy_m[i] = self.policy[st][act]
                i += 1
        return policy_m
    
    def dh_dx(self):
        #depends on the definition of h(x)
        #This is a linear sum of x
        grad = np.zeros(self.x_size)
        for m in self.modify_list:
            grad[m] = self.weight
        return -grad
    
    def dJ_dtheta(self, Sample):
        #grdient of value function respect to theta
        #sample based method
        #returns dJ_dtheta_i, 1*NM matrix
        N = len(Sample.trajlist)
        grad = 0
        for rho in Sample.trajlist:
            grad += self.drho_dtheta(rho) * self.mdp.reward_traj(rho, 0)
        return 1/N * grad
    
    def drho_dtheta(self, rho):
        if len(rho) == 1:
            return np.zeros(self.x_size)
        st = rho[0]
        act = rho[1]
        rho = rho[2:]
        return self.dPi_dtheta(st, act) + self.drho_dtheta(rho)
    
    def dPi_dtheta(self, st, act):
        #dlog(pi)_dtheta
        grad = np.zeros(self.x_size)
        st_index = self.mdp.states.index(st)
        act_index = self.mdp.actions.index(act)
        Pi = self.policy[st]
        for i in range(self.act_len):
            if i == act_index:
                grad[st_index * self.act_len + act_index] = 1/self.tau * (1 - Pi[act])
            else:
                grad[st_index * self.act_len + act_index] = 1/self.tau * (0 - Pi[act])
        #grad is a vector x_size * 1
        return grad
            
    def dtheta_dx(self):
        #returns a NM X NM matrix, (i, j) is dtheta_i/dx_j
        grad = np.zeros((self.x_size, self.x_size))
        for m in self.modify_list:
            grad_l = self.dtheta_dx_line(m)
            grad[:, m] = grad_l
        return grad
    
    def dtheta_dx_line(self, index):
        #dtheta_dx(s, a) = dtheta_dr(s, a) * dr(s, a)_dx(s, a), dr(s, a)_dx(s, a) = 1
        #what we realize is dtheta_dr(s, a) here
        #dtheta_dx_line returns one column in the dtheta_dx matrix
        dtheta = np.zeros(self.x_size)
        r_indicator = np.zeros(self.x_size)
        r_indicator[index] = 1
        dtheta_old = dtheta.copy()
        delta = np.inf
        while delta > self.epsilon:
            dtheta = r_indicator + self.mdp.gamma * self.P_matrix.dot(self.policy_m * dtheta)
            delta = np.max(abs(dtheta - dtheta_old))
            dtheta_old = dtheta
        return dtheta
        
    
    def dJ_dx(self):
        dh_dx = self.dh_dx()
        dJ_dtheta = self.dJ_dtheta(self.sample)
        dtheta_dx = self.dtheta_dx()
        dJ_dx = dJ_dtheta * dtheta_dx + dh_dx
        self.update_x(dJ_dx)
        
        
    def update_x(self, gradient):
        self.x += self.lr_x * gradient
        
    def update_policy(self, policy):
        self.policy = policy_convert(policy, self.mdp.actions)
        self.policy_m = self.convert_policy()
        self.sample.generate_traj(100, self.policy)
        
    def SGD(self, epsilon):
        delta = np.inf
        J_old, policy = self.J_func()
        self.update_policy(policys)
        while delta > epsilon:
            self.dJ_dx()
            J_new, policy = self.J_func()
            #update it to new policy
            self.update_policy(policy)
            delta = abs(J_new - J_old)

def policy_convert(pi, action_list):
    #Convert a policy from pi[st][act] = pro to pi[st] = [pro1, pro2, ...]
    pi_list = {}
    for st in pi.keys():
        pro = []
        for act in action_list:
            pro.append(pi[st][act])
        pi_list[st] = pro
    return pi_list

def reward2list(reward, states, actions):
    reward_s = []
    for st in states:
        for act in actions:
            reward_s.append(reward[st][act])
    return np.array(reward_s)

if __name__ == "__main__":
    mdp = MDP.create_mdp()
    V, policy = mdp.get_policy_entropy([])
    lr_x = 0.01   #The learning rate of side-payment
    modifylist = [48]  #The action reward you can modify
    epsilon = 0.001   #Convergence threshold
    weight = 0.1