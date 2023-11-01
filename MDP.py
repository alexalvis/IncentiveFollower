# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:38:49 2023

@author: hma2
"""

import numpy as np

class MDP:
    def __init__(self, gamma, tau):
        self.states = self.getstates()
        self.actions = self.getactions()
        self.F = self.getfakegoals()
        self.G = self.getgoals()
        self.IDS = self.getIDS()
        self.init = self.getInit()
        self.transition = self.gettransition()
        self.update_reward()
        self.reward_l = self.leader_reward()
        self.gamma = gamma
        self.tau = tau
        self.nextSt_list, self.nextPro_list = self.stotrans_list()
    
    def getstates(self):
        states = [
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
        ]
        return states
        
    def getactions(self):
        A = ["a", "b", "c", "d"]
        return A
    
    def getfakegoals(self):
        F = ["q11", "q12"]
        return F

    def getgoals(self):
        G = ["q10"]
        return G
    
    def getIDS(self):
        IDS = ["q5", "q8"]
        return IDS
    
    def getInit(self):
        init = np.zeros(len(self.states))
        init[0] = 1
        return init
    
    def gettransition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
        trans["q0"]["a"] = "q1"
        trans["q0"]["b"] = "q2"
        trans["q0"]["c"] = "q3"
        trans["q0"]["d"] = "q4"
        trans["q1"]["a"] = "q5"
        trans["q1"]["b"] = "q8"
        trans["q1"]["c"] = "q6"
        trans["q2"]["a"] = "q6"
        trans["q2"]["b"] = "q7"
        trans["q3"]["b"] = "q5"
        trans["q3"]["c"] = "q7"
        trans["q4"]["c"] = "q7"
        trans["q4"]["d"] = "q5"
        trans["q5"]["a"] = "q5"
        trans["q5"]["b"] = "q5"
        trans["q5"]["c"] = "q5"
        trans["q5"]["d"] = "q5"
        trans["q6"]["b"] = "q9"
        trans["q6"]["d"] = "q10"
        trans["q7"]["a"] = "q9"
        trans["q7"]["b"] = "q8"
        trans["q8"]["a"] = "q8"
        trans["q8"]["b"] = "q8"
        trans["q8"]["c"] = "q8"
        trans["q8"]["d"] = "q8"
        trans["q9"]["b"] = "q11"
        trans["q9"]["c"] = "q12"
        trans["q10"]["a"] = "q10"
        trans["q10"]["b"] = "q10"
        trans["q10"]["c"] = "q10"
        trans["q10"]["d"] = "q10"
        trans["q11"]["a"] = "q11"
        trans["q11"]["b"] = "q11"
        trans["q11"]["c"] = "q11"
        trans["q11"]["d"] = "q11"
        trans["q12"]["a"] = "q12"
        trans["q12"]["b"] = "q12"
        trans["q12"]["c"] = "q12"
        trans["q12"]["d"] = "q12"
        stotrans = self.stochastictrans(trans)
        return stotrans
    
    def stochastictrans(self, trans):
        stotrans = {}
        for st in self.states:
            if (st not in self.F) and (st not in self.G) and (st not in self.IDS):
                stotrans[st] = {}
                for act in self.actions:
                    stotrans[st][act] = {}
                    if act in trans[st].keys():
                        stotrans[st][act][st] = 0
                        stotrans[st][act][trans[st][act]] = 0.7
                        for otheract in self.actions:
                            if otheract != act:
                                if otheract not in trans[st].keys():
                                    stotrans[st][act][st] += 0.1
                                else:
                                    if trans[st][otheract] not in stotrans[st][act].keys():
                                        stotrans[st][act][trans[st][otheract]] = 0.1
                                    else:
                                        stotrans[st][act][trans[st][otheract]] += 0.1
                    else:
                        stotrans[st][act][st] = 0.7
                        for otheract in self.actions:
                            if otheract != act:
                                if otheract not in trans[st].keys():
                                    stotrans[st][act][st] += 0.1
                                else:
                                    if trans[st][otheract] not in stotrans[st][act].keys():
                                        stotrans[st][act][trans[st][otheract]] = 0.1
                                    else:
                                        stotrans[st][act][trans[st][otheract]] += 0.1
            else:
                stotrans[st] = {}
                for act in self.actions:
                    stotrans[st][act] = {}
                    stotrans[st][act]["Sink"] = 1.0
        
        if checkstotrans(stotrans):
            return stotrans
        else:
            print(stotrans)
    
    def init_value(self):
        return np.zeros(len(self.states))
    
    def update_reward(self, reward = []):
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
    
    def initial_reward(self):
        self.reward = {}
        for st in self.states:
            self.reward[st] = {}
            if st in self.G:
                for act in self.actions:
                    self.reward[st][act] = 1.0
            else:
                for act in self.actions:
                    self.reward[st][act] = 0.0
    
    def leader_reward(self):
        leader_reward = {}
        for st in self.states:
            self.reward[st] = {}
            if st in self.F:
                for act in self.actions:
                    self.reward[st][act] = 1.0
            else:
                for act in self.actions:
                    self.reward[st][act] = 0.0
        return leader_reward
        
        
    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.transition[st][act].items():
            if st_ != "Sink":
                core += pro * V[self.states.index(st_)]
        return core
    
    def get_policy_entropy(self, reward):
        threshold = 0.0001
        self.update_reward(reward)
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
                for act in self.actions:
                    core = (self.reward[st][act] + self.gamma * self.getcore(V1, st, act)) / self.tau
                    Q[st][act] = np.exp(core)
                Q_s = sum(Q[st].values())
                for act in self.actions:
                    policy[st][act] = Q[st][act] / Q_s
                V[self.states.index(st)] = self.tau * np.log(Q_s)
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
        traj = []
        st = np.random.choice(self.states, 1, p = self.init)[0]
        # st_index = self.states.index(st)
        act = np.random.choice(self.actions, 1, p = pi[st])[0]
        traj.append(st)
        traj.append(act)
        next_st = self.one_step_transition(st, act)
        while next_st != "Sink":
            st = next_st
            st_index = self.states.index(st)
            act = np.random.choice(self.actions, 1, p = pi[st])[0]
            traj.append(st)
            traj.append(act)
            next_st = self.one_step_transition(st, act)
        traj.append(next_st)
        return traj
    
    def one_step_transition(self, st, act):
        st_list = self.nextSt_list[st][act]
        pro_list = self.nextPro_list[st][act]
        next_st = np.random.choice(st_list, 1, p = pro_list)[0]
        return next_st
        
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

def checkstotrans(trans):
    for st in trans.keys():
        for act in trans[st].keys():
            if abs(sum(trans[st][act].values()) - 1) > 0.01:
                print(
                    "st is:",
                    st,
                    " act is:",
                    act,
                    " sum is:",
                    sum(trans[st][act].values()),
                )
                return False
    return True

def policy_convert(pi, action_list):
    #Convert a policy from pi[st][act] = pro to pi[st] = [pro1, pro2, ...]
    pi_list = {}
    for st in pi.keys():
        pro = []
        for act in action_list:
            pro.append(pi[st][act])
        pi_list[st] = pro
    return pi_list

def create_mdp():
    gamma = 0.95
    tau = 0.01
    mdp = MDP(gamma, tau)
    return mdp

if __name__ == "__main__":
    mdp = create_mdp()
    V, policy = mdp.get_policy_entropy([])