# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:30:05 2023

@author: hma2
"""

import numpy as np
import MDP


class SampleTraj:
    def __init__(self, mdp):
        self.mdp = mdp
        self.trajlist = []
    
    def generate_traj(self, N, policy):
        self.reset()
        for i in range(N):
            traj  = self.mdp.generate_sample(policy)
            self.trajlist.append(traj)
    
    def reset(self):
        #Reset the generated samples
        self.trajlist = []

    def approx_policy(self):
        st_count = {st: 0 for st in self.mdp.states}
        st_act_count = {}
        for st in self.mdp.states:
            st_act_count[st] = {}
            for act in self.mdp.actions:
                st_act_count[st][act] = 0

        for traj in self.trajlist:
            st = traj[0]
            act = traj[1]
            while len(traj) >=3:
                st_count[st] += 1
                st_act_count[st][act] += 1
                traj = traj[2:]
        policy = {}
        for st in self.mdp.states:
            policy[st] = {}
            if st_count[st] == 0:
                for act in self.mdp.actions:
                    policy[st][act] = 0
            else:
                for act in self.mdp.actions:
                    policy[st][act] = st_act_count[st][act] / st_count[st]
        return policy


def test():
    mdp = MDP.create_mdp()
    sample = SampleTraj(mdp)
    V, policy = mdp.get_policy_entropy([])
    policy_c = MDP.policy_convert(policy, mdp.actions)
    print(policy_c)
    sample.generate_traj(100, policy_c)
    return sample
    
    
if __name__ == "__main__":
    sample = test()
    print(sample.trajlist[0])
