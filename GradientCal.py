# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:47:47 2023

@author: hma2
"""

import numpy as np
import MDP
import math
import Sample
import GridWorld as GW
import copy

class GC:
    def __init__(self, mdp, lr_x, policy, epsilon, modify_list, weight, approximate_flag):
        self.mdp = mdp
        self.st_len = len(mdp.states)
        self.act_len = len(mdp.actions)
        self.x_size = self.st_len * self.act_len
        self.base_reward = reward2list(mdp.chef_reward, mdp.states, mdp.actions)
        self.x = np.zeros(self.x_size)
        self.lr_x = lr_x
        self.tau = self.mdp.tau
        self.policy = policy_convert(policy, mdp.actions)           #self.policy in the form of pi[st]= [pro1, pro2, ...]   #Leader's perspective policy
        self.policy_m = self.convert_policy(policy)          #self.policy_m is a vector.
        self.P_matrix = self.construct_P()
        self.epsilon = epsilon
        self.modify_list = modify_list
        self.weight = weight
        self.sample = Sample.SampleTraj(self.mdp)
        self.approximate_flag = approximate_flag
        #0 is not using approximate_policy, 1 is using approximate_policy
        
    def reward_sidepay(self):
        return self.base_reward + self.x
    
    def J_func(self):
        reward_s = self.reward_sidepay()
        V, policy = self.mdp.get_policy_entropy(reward_s, flag = 1)
        
        #Need to evaluate defender's reward using this policy
        V_leader = self.mdp.policy_evaluation([], 0, policy)
        J = self.mdp.init.dot(V_leader) + self.h_func()
        return J, policy

    def h_func(self):
        #Define the h function
        return -self.weight * np.sum(abs(self.x))
    
    def construct_P(self):
        P = np.zeros((self.x_size, self.x_size))
        for i in range(self.st_len):
            for j in range(self.act_len):
                for next_st, pro in self.mdp.transition[self.mdp.states[i]][self.mdp.actions[j]].items():
                    if next_st != 'Sink':
                        next_index = self.mdp.states.index(next_st)
                        P[i * self.act_len + j][next_index * self.act_len : (next_index + 1) * self.act_len] = pro
        return P
    
    def convert_policy(self, policy):
        policy_m = np.zeros(self.x_size)
        i = 0
        for st in self.mdp.states:
            for act in self.mdp.actions:
                policy_m[i] = policy[st][act]
                i += 1
        return policy_m
    
    def dh_dx(self):
        #depends on the definition of h(x)
        #This is a linear sum of x
        grad = np.zeros(self.x_size)
        for m in self.modify_list:
            if self.x[m] >= 0:
                grad[m] = self.weight
            else:
                self.x[m] = -self.weight
        return -grad
    
    def dJ_dtheta(self, Sample):
        #grdient of value function respect to theta
        #sample based method
        #returns dJ_dtheta_i, 1*NM matrix
        N = len(Sample.trajlist)
        grad = 0
        for rho in Sample.trajlist:
            # print("trajectory is:", rho)
            grad += self.drho_dtheta(rho) * self.mdp.reward_traj(rho, 0)
            # print(self.drho_dtheta(rho))
        # print("grad is:", grad)
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
        # print("Pi:", Pi)
        for i in range(self.act_len):
            if i == act_index:
                grad[st_index * self.act_len + i] = 1/self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * self.act_len + i] = 1/self.tau * (0.0 - Pi[i])
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
        itcount_d = 0
        while delta > self.epsilon:
            # print(f"{itcount_d} iterations")
            # print(self.policy_m)
            # print(self.policy_m * dtheta)
            dtheta = r_indicator + self.mdp.gamma * self.P_matrix.dot(self.policy_m * dtheta)
            delta = np.max(abs(dtheta - dtheta_old))
            dtheta_old = dtheta
            itcount_d += 1
        # dtheta_ = self.mdp.theta_evaluation(r_indicator, self.policy)
        # print("x is", self.x)
        # print("Matrix_result:", dtheta)
        # print("Evaluation result:", dtheta_)
        return dtheta
        
    
    def dJ_dx(self, N, policy):
        dh_dx = self.dh_dx()
        # print(dh_dx)
        # print(policy)
        self.sample.generate_traj(N, policy)
        if self.approximate_flag:
            self.policy = policy_convert(self.sample.approx_policy(), self.mdp.actions)
        # print("approximate policy is:", self.policy)

        dJ_dtheta = self.dJ_dtheta(self.sample)
        # print("dJ_dtheta:", dJ_dtheta)
        # print(dJ_dtheta)
        dtheta_dx = self.dtheta_dx()
        # print("dtheta_dx:", dtheta_dx[:,48])
        dJ_dtheta_x = dJ_dtheta.dot(dtheta_dx)
        dJ_dx = dJ_dtheta_x + dh_dx
        # print("dJ_dx:", dJ_dx)
        self.update_x(dJ_dx)
        
        
    def update_x(self, gradient):
        self.x += self.lr_x * gradient
        self.x = np.maximum(self.x, 0)
        
    def update_policy(self, policy, N):
        # if self.approximate_flag:
        #     Leader uses approximate policy
            # policy = policy_convert(policy, self.mdp.actions)
            # self.sample.generate_traj(N, policy)
            # policy_ = self.sample.approx_policy()
            # self.policy_m = self.convert_policy(policy_)
            # self.policy = policy_convert(policy_, self.mdp.actions)
        # else:
            #Leader uses exact policy
        self.policy_m = self.convert_policy(policy)
        self.policy = policy_convert(policy, self.mdp.actions)
        self.sample.generate_traj(N, self.policy)
        
    def SGD(self, N, max_iterations=-1):
        x_history = []
        J_history = []
        # Append initial value of weights
        x_history.append(tuple(self.x))

        delta = np.inf
        J_old, policy = self.J_func()   #policy is exact policy
        policy_c = policy_convert(policy, self.mdp.actions)   #exact policy
        self.update_policy(policy, N)   #exact or approximate policy, depends on flag
        itcount = 1

        # changed_set = set()
        while delta > self.epsilon:
            x_history.append(tuple(self.x))
            J_history.append(J_old)
            self.dJ_dx(N, policy_c)    #
            J_new, policy = self.J_func()   # exact policy
            policy_c = policy_convert(policy, self.mdp.actions)   #exact policy
            print("itcount", itcount)
            print("J_new:", J_new)
            #update it to new policy
            self.update_policy(policy, N)
            delta = abs(J_new - J_old)
            print("delta:", delta)
            J_old = J_new
            # print(f"{itcount}th iteration")

            # non_zero = [i for i, n in enumerate(self.x) if n != 0]
            #
            # changed_set.update(non_zero)
            # print(changed_set)

            # Print state_action pairs with non-zero side payment
            # for i, side_payment in enumerate(self.x):
            #     if side_payment != 0:
            #         print(f"{self.mdp.x_index_2_state_action(i)}: {side_payment}")
            # print(self.x)
            itcount += 1
            if itcount % 100 == 0:
                print(f'{itcount}th iteration, x is {self.x}')
            if max_iterations != -1 and itcount >= max_iterations:
                print(f"Stopping at itcount {itcount}")
                break
        return self.x, x_history, J_history

    def get_reward(self):
        i = 0
        reward = dict(zip(self.mdp.states, [dict(zip(self.mdp.actions, [0 for _ in self.mdp.actions])) for _ in self.mdp.states]))
        for st in self.mdp.states:
            for act in self.mdp.actions:
                reward[st][act] = self.reward_sidepay()[i]
                i += 1
        return reward
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

def MDP_example():
    #This is the function used to generate small transition MDP example.
    mdp = MDP.create_mdp()
    V, policy = mdp.get_policy_entropy([], 1)
    #Learning rate influence the result from the convergence aspect. Small learning rate wll make the convergence criteria satisfy too early.
    lr_x = 0.02 #The learning rate of side-payment
    modifylist = [48]  #The action reward you can modify
    epsilon = 1e-6   #Convergence threshold
    weight = 0  #weight of the cost
    approximate_flag = 0  #Whether we use trajectory to approximate policy. 0 represents exact policy, 1 represents approximate policy
    GradientCal = GC(mdp, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    x_res = GradientCal.SGD(N = 50)
    print(x_res)

def GridW_example():
    mdp = GW.createGridWorldBarrier_new2()
    V, policy = mdp.get_policy_entropy([], 1)
    lr_x = 0.01
    modifylist = [40, 116]
    epsilon = 1e-6
    weight = 0
    approximate_flag = 1
    GradientCal = GC(mdp, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    x_res = GradientCal.SGD(N = 200)
    print(x_res)

if __name__ == "__main__":
    MDP_example()
    # GridW_example()