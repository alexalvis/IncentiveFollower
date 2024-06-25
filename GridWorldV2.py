#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:05:02 2024

@author: hma
"""

# -*- coding: utf-8 -*-

import numpy as np
import copy

class GridWorld:
    def __init__(self, width, height, stoPar, F, G, IDS, Barrier, gamma, tau):
        self.width = width
        self.height = height
        self.stoPar = stoPar
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)] #E, W, S, N
        self.complementA = self.getComplementA()
        self.states = self.getstate()
        self.addBarrier(Barrier)
        self.F = F        
        self.G = G           
        self.IDS = IDS
        self.init = self.getInit()
        self.transition = self.gettrans()
        self.update_reward()
        self.reward_l = self.leader_reward()
        self.gamma = gamma
        self.tau = tau
        self.nextSt_list, self.nextPro_list = self.stotrans_list()

    def getInit(self):
        I = np.zeros(len(self.states))
        I[12] = 1
        return I
    
    def getstate(self):
        states = []
        for i in range(self.width):
            for j in range(self.height):
                states.append((i, j))
        return states
    
    def checkinside(self, st):
        if st in self.states:
            return True
        return False
    
    def getComplementA(self):
        complementA = {}
        complementA[(0, 1)] = [(1, 0), (-1, 0)]
        complementA[(0, -1)] = [(1, 0), (-1, 0)]
        complementA[(1, 0)] = [(0, 1), (0, -1)]
        complementA[(-1, 0)] = [(0, 1), (0, -1)]
        return complementA
        
    def gettrans(self):
        #Calculate transition
        stoPar = self.stoPar
        trans = {}
        for st in self.states:
            trans[st] = {}
            if (st not in self.F) and (st not in self.G):
                for act in self.actions:
                    trans[st][act] = {}
                    trans[st][act][st] = 0
                    tempst = tuple(np.array(st) + np.array(act))
                    if self.checkinside(tempst):
                        trans[st][act][tempst] = 1 - 2*stoPar
                    else:
                        trans[st][act][st] += 1- 2*stoPar
                    for act_ in self.complementA[act]:
                        tempst_ = tuple(np.array(st) + np.array(act_))
                        if self.checkinside(tempst_):
                            trans[st][act][tempst_] = stoPar
                        else:
                            trans[st][act][st] += stoPar
            else:
                for act in self.actions:
                    trans[st][act] = {}
                    trans[st][act]["Sink"] = 1.0
        
        if self.checktrans(trans):
            return trans
        else:
            print("Transition is incorrect")

    
    def checktrans(self, trans):
        for st in self.states:
            for act in self.actions:
                if abs(sum(trans[st][act].values())-1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(trans[st][act].values()))
                    return False
        return True

        
    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        for st in Barrierlist:
            self.states.remove(st)
    
    def init_value(self):
        #Initial the value to be all 0
        return np.zeros(len(self.states))
    
    def update_reward(self, reward = []):
        #Update follower's reward
        if len(reward) >0:
            self.reward = {}
            i = 0
            for st in self.states:
                self.reward[st] = {}
                for act in self.actions:
                    self.reward[st][act] = reward[i]
                    i += 1
        else:
            self.initial_reward()
        
    def leader_reward(self):
        leader_reward = {}
        for st in self.states:
            leader_reward[st] = {}
            if st in self.F:
                for act in self.actions:
                    leader_reward[st][act] = 10.0
            else:
                for act in self.actions:
                    leader_reward[st][act] = 0.0
        return leader_reward    
            
    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.transition[st][act].items():
            if st_ != "Sink":
                core += pro * V[self.states.index(st_)]
        return core
    
    def initial_reward(self):
        self.reward = {}
        for st in self.states:
            self.reward[st] = {}
            if st in self.G:
                for act in self.actions:
                    self.reward[st][act] = 10.0
            elif st in self.IDS:
                for act in self.actions:
                    self.reward[st][act] = -5.0
            elif st in self.F:
                for act in self.actions:
                    self.reward[st][act] = 8.0
            else:
                for act in self.actions:
                    self.reward[st][act] = 0
                
    
    def policy_evaluation(self, reward, flag, policy):
        threshold = 0.00001
        if flag == 0:
            reward = self.reward_l
        else:
            self.update_reward(reward)
            reward = self.reward
        V = self.init_value()
        delta = np.inf
        while delta > threshold:
            V1 = V.copy()
            for st in self.states:
                temp = 0
                for act in self.actions:
                    if act in policy[st].keys():
                        temp += policy[st][act] * (reward[st][act] + self.gamma * self.getcore(V1, st, act))
                V[self.states.index(st)] = temp
            delta = np.max(abs(V-V1))
        return V
    
    def get_policy_entropy(self, reward, flag):
        threshold = 0.0001
        if flag == 0:
            reward = self.reward_l
        else:
            self.update_reward(reward)
            reward = self.reward
        V = self.init_value()
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.states:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.states:
                Q_theta = []
                for act in self.actions:
                    core = (reward[st][act] + self.gamma * self.getcore(V1, st, act)) / self.tau
                    # Q[st][act] = np.exp(core)
                    Q_theta.append(core)
                Q_sub = Q_theta - np.max(Q_theta)
                p = np.exp(Q_sub)/np.exp(Q_sub).sum()
                # Q_s = sum(Q[st].values())
                # for act in self.actions:
                    # policy[st][act] = Q[st][act] / Q_s
                for i in range(len(self.actions)):
                    policy[st][self.actions[i]] = p[i]
                V[self.states.index(st)] = self.tau * np.log(np.exp(Q_theta).sum())
            itcount += 1
        return V, policy
    
    def stotrans_list(self):
        transition_list = {}
        transition_pro = {}
        for st in self.transition:
            transition_list[st] = {}
            transition_pro[st] = {}
            for act in self.transition[st]:
                transition_list[st][act] = {}
                transition_pro[st][act] = {}
                st_list = []
                pro_list = []
                for st_, pro in self.transition[st][act].items():
                    st_list.append(st_)
                    pro_list.append(pro)
                transition_list[st][act] = st_list
                transition_pro[st][act] = pro_list
        return transition_list, transition_pro
    
    def generate_sample(self, pi):
        #pi here should be pi[st] = [pro1, pro2, ...]
        traj = []
        st_index = np.random.choice(len(self.states), 1, p = self.init)[0]
        st = self.states[st_index]
        act_index = np.random.choice(len(self.actions), 1, p = pi[st])[0]
        act = self.actions[act_index]
        traj.append(st)
        traj.append(act)
        next_st = self.one_step_transition(st, act)
        while next_st != "Sink":
            st = next_st
            # st_index = self.states.index(st)
            act_index = np.random.choice(len(self.actions), 1, p = pi[st])[0]
            act = self.actions[act_index]
            traj.append(st)
            traj.append(act)
            next_st = self.one_step_transition(st, act)
        traj.append(next_st)
        return traj
    
    def one_step_transition(self, st, act):
        st_list = self.nextSt_list[st][act]
        pro_list = self.nextPro_list[st][act]
        next_st = np.random.choice(len(st_list), 1, p = pro_list)[0]
        return st_list[next_st]
        
    def reward_traj(self, traj, flag):
        #Flag is used to identify whether it is leader's reward or follower
        #Flag = 0 represents leader, Flag = 1 represents follower
        if flag == 0:
            reward = self.reward_l
        else:
            reward = self.reward
        st = traj[0]
        act = traj[1]
        if len(traj) >= 4:
            r = reward[st][act] + self.gamma * self.reward_traj(traj[2:], flag)
        else:
            return reward[st][act]
        return r
    
    def stVisitFre(self, policy):
        threshold = 0.00001
        gamma = 0.95
        Z0 = np.zeros(len(self.states))
#        Z0[9] = 1
        # Z0[12] = 1  #6*6 case   #12 corresponds to the scenario in ppt
#        Z0[51] = 1  #10*10 case
        Z0[51] = 1 #10*10 case
        Z_new = Z0.copy()
        Z_old = Z_new.copy()
        itcount = 1
#        sinkst = self.F + self.G + self.IDS
#        print(sinkst)
        while itcount == 1 or np.inner(np.array(Z_new)-np.array(Z_old), np.array(Z_new)-np.array(Z_old)) > threshold:
#            print(itcount)
            Z_old = Z_new.copy()
            Z_new = Z0.copy()
            for st in self.states:
                index_st = self.states.index(st)
                for act in self.actions:
                    for st_ in self.states:
                        if st in self.transition[st_][act].keys():
                            Z_new[index_st] += gamma * Z_old[self.states.index(st_)] * policy[st_][act] * self.transition[st_][act][st]
            
            itcount += 1
#            print(Z)
#            diff = np.subtract(Z_new, Z_old)
#            diff_list = list(diff)
#            print(diff_list)
        return Z_new
    
    def stactVisitFre(self, policy):
        Z = self.stVisitFre(policy)
        st_act_visit = {}
        for i in range(len(self.states)):
            st_act_visit[self.states[i]] ={}
            for act in self.actions:
                st_act_visit[self.states[i]][act] = Z[i] * policy[self.states[i]][act]
        return st_act_visit

def createGridWorldBarrier_new2():
    gamma = 0.95
    tau = 0.1
    goallist = [(3, 4), (5, 0)] 
    barrierlist = []
    fakelist = [(1, 4), (4, 5)] #case 1
    # fakelist = [(0, 2), (5, 3)] #case 2
    IDSlist = [(0, 4), (1, 2), (2, 3), (3, 3), (5, 4)]
    gridworld = GridWorld(6, 6, 0.1, fakelist, goallist, IDSlist, barrierlist, gamma, tau)
    reward = []
    # V, policy = gridworld.get_policy_entropy(reward, 1)
    # V_def = gridworld.policy_evaluation(reward, 0, policy)
    # return gridworld, V_def, policy
    return gridworld

    
    
if __name__ == "__main__":
    # gridworld, V_def, policy = createGridWorldBarrier_new2()
    gridworld = createGridWorldBarrier_new2()
    
    # print(gridworld.reward)
    # for st in gridworld.F:
    #     print(gridworld.states.index(st))
    # Z = gridworld.stVisitFre(policy)
    # Z_act = gridworld.stactVisitFre(policy)
    # print(V_def[51])
#    print(V_def[14], Z[20], Z[48])
#    print(Z[35], Z[54])