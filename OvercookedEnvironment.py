# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:27:12 2023

@author: hma2
"""

import numpy as np
import itertools
import copy


class OvercookedEnvironment:
    def __init__(self, stove_size, counter_size, burn_probability, cold_probability, deliver_probability,
                 cook_food_reward, burned_food_penalty, warm_food_delivered_reward, cold_food_delivered_reward):
        """
        :param stove_size:
        :param counter_size:
        :param burn_probability: Probability of a cooking food transitioning to burned between states
        :param cold_probability: Probability of a warm food transitioning to cold between states
        :param deliver_probability: Probability of the waiter delivering a dish between states
        :param cook_food_reward: Reward given when chef moves cooked food to counter
        :param burned_food_penalty: Negative reward for chef when burned food is on the stove
        :param warm_food_delivered_reward: Leader's reward when warm food is delivered
        :param cold_food_delivered_reward: Leader's reward when cold food is delivered
        """
        self.stove_size = stove_size
        self.counter_size = counter_size

        self.burn_probability = burn_probability
        self.cold_probability = cold_probability
        self.deliver_probability = deliver_probability

        self.cook_food_reward = cook_food_reward
        self.burned_food_penalty = burned_food_penalty
        self.warm_food_delivered_reward = warm_food_delivered_reward
        self.cold_food_delivered_reward = cold_food_delivered_reward

        self.stove_states = self.get_stove_states()
        self.counter_states = self.get_counter_states()
        self.states = self.get_states()

        self.actions = self.get_actions()
        self.transition = self.get_transition()

    def get_states(self):
        """
        :return: The cartesian product of the set of all possible stove and counter states.

        States are in the form:
        state=((stove_state), (counter_state))
        """
        states = list(itertools.product(self.stove_states, self.counter_states))
        return states

    def get_stove_states(self):
        """
        :return: A list of all possible stove states. stove_states is the cartesian product of all stove conditions.
        """
        stove_conditions = ["empty", "cooking", "burned"]
        stove_states = list(itertools.product(stove_conditions, repeat=self.stove_size))
        return stove_states

    def get_counter_states(self):
        """
        :return: A list of all possible counter states. counter_states is the cartesian product of all counter conditions.
        """
        counter_conditions = ["empty", "warm", "cold"]
        counter_states = list(itertools.product(counter_conditions, repeat=self.counter_size))
        return counter_states

    @staticmethod
    def get_actions():
        actions = ["wait", "add_food", "move_food", "deliver", "clear_burned_food"]
        return actions

    def get_transition(self):
        """
        :return: Transition function is the form of trans[state][action][next_state] = probability. This function is
        defined for all state-action pairs.
        """
        # Define transitions
        trans = dict(zip(self.states, [dict(zip(self.actions, [dict() for _ in self.actions])) for _ in self.states]))
        for state in self.states:
            stove_state = state[0]
            counter_state = state[1]
            for action in self.actions:
                if action == "wait":
                    possible_states = self.get_wait_states(state)
                    for new_state in possible_states:
                        trans[state][action][new_state] = OvercookedEnvironment.get_transition_probability(
                            state, new_state, self.burn_probability, self.cold_probability, self.deliver_probability)
                elif action == "add_food":
                    # in the case where there are no empty burners the transition is the same as wait
                    possible_states = self.get_wait_states(state)
                    for new_state in possible_states:
                        new_stove_state = new_state[0]
                        # in the case where there is an open burner the transitions are the same as wait but one of the
                        # "empty" burner's becomes "cooking"
                        if "empty" in new_stove_state:
                            stove_state_with_added_food = list(new_stove_state)
                            for i, burner in enumerate(stove_state_with_added_food):
                                if burner == "empty":
                                    stove_state_with_added_food[i] = "cooking"
                                    break
                            state_with_food_added = (tuple(stove_state_with_added_food), new_state[1])
                            trans[state][action][state_with_food_added] = \
                                OvercookedEnvironment.get_transition_probability(state, new_state,
                                                                                 self.burn_probability,
                                                                                 self.cold_probability,
                                                                                 self.deliver_probability)

                        # if there is no empty burner the transition is the same as "wait"
                        else:
                            trans[state][action][new_state] = \
                                OvercookedEnvironment.get_transition_probability(state, new_state,
                                                                                 self.burn_probability,
                                                                                 self.cold_probability,
                                                                                 self.deliver_probability)
                elif action == "move_food":
                    origin_state_with_food_moved = OvercookedEnvironment.get_state_after_move_food(state)
                    possible_states = self.get_wait_states(origin_state_with_food_moved)
                    for new_state in possible_states:
                        trans[state][action][new_state] = \
                            OvercookedEnvironment.get_transition_probability(origin_state_with_food_moved, new_state,
                                                                             self.burn_probability,
                                                                             self.cold_probability,
                                                                             self.deliver_probability)
                elif action == "deliver":
                    origin_state_without_plate = OvercookedEnvironment.get_state_after_deliver(state)
                    possible_states = self.get_wait_states(origin_state_without_plate)
                    for new_state in possible_states:
                        trans[state][action][new_state] = \
                            OvercookedEnvironment.get_transition_probability(origin_state_without_plate, new_state,
                                                                             self.burn_probability,
                                                                             self.cold_probability,
                                                                             self.deliver_probability)
                elif action == "clear_burned_food":
                    origin_state_without_burned_food = OvercookedEnvironment.get_state_after_remove_burned(state)
                    possible_states = self.get_wait_states(origin_state_without_burned_food)
                    for new_state in possible_states:
                        trans[state][action][new_state] = \
                            OvercookedEnvironment.get_transition_probability(origin_state_without_burned_food,
                                                                             new_state,
                                                                             self.burn_probability,
                                                                             self.cold_probability,
                                                                             self.deliver_probability)
        self.check_trans(trans)
        return trans

    @staticmethod
    def get_state_after_remove_burned(state):
        stove_state = state[0]
        new_stove_state = list(stove_state)
        if "burned" in stove_state:
            for i, burner in enumerate(new_stove_state):
                if burner == "burned":
                    new_stove_state[i] = "empty"
                    break
        return new_stove_state

    @staticmethod
    def get_state_after_move_food(state):
        """
        :param state: A state in the form (stove_state, counter_state)
        :return: State tuple with food moved from the stove to the counter (ie if there is something "cooking" in stove
        state and "empty" in counter_state the "cooking" will be set to "empty" and the "empty" set to "warm"
        """
        stove_state = state[0]
        counter_state = state[1]

        new_stove_state = list(stove_state)
        new_counter_state = list(counter_state)
        if "cooking" in stove_state and "empty" in counter_state:
            # remove food from stove
            for i, burner in enumerate(new_stove_state):
                if burner == "cooking":
                    new_stove_state[i] = "empty"
                    break
            # add food to counter
            for i, plate in enumerate(new_counter_state):
                if plate == "empty":
                    new_counter_state[i] = "warm"
                    break
        state_with_food_moved = (tuple(new_stove_state), tuple(new_counter_state))
        return state_with_food_moved

    @staticmethod
    def get_state_after_deliver(state):
        """
        :param state: An original state to remove a dish from
        :return: The new state after removing a dish from the counter with priority to cold dishes
        """
        counter_state = state[1]
        new_counter_state = list(counter_state)
        if "cold" in counter_state:
            for i, plate in enumerate(new_counter_state):
                if plate == "cold":
                    new_counter_state[i] = "empty"
                    break
        elif "warm" in counter_state:
            for i, plate in enumerate(new_counter_state):
                if plate == "warm":
                    new_counter_state[i] = "empty"
                    break
        new_state = (state[0], tuple(new_counter_state))
        return new_state

    @staticmethod
    def get_transition_probability(old_state, new_state, burn_probability, cold_probability, deliver_probability):
        """
        :param old_state: The origin state
        :param new_state: The target state that is being transitioned to
        :param burn_probability: The probability of each "cooking" burner to transition to burned
        :param cold_probability: The probability of each "warm" plate to transition to "cold"
        :param deliver_probability: The probability of a plate being passively delivered by the waiter
        :return: The probability of transitioning old_state -> new_state if no action (wait) is taken
        """
        old_stove_state = old_state[0]
        old_counter_state = old_state[1]

        new_stove_state = new_state[0]
        new_counter_state = new_state[1]

        old_cooking_total, old_burned_total, old_empty_burner_total = OvercookedEnvironment.get_stove_state_counts(
            old_stove_state)
        new_cooking_total, new_burned_total, new_empty_burner_total = OvercookedEnvironment.get_stove_state_counts(
            new_stove_state)

        food_available_for_delivery = sum([1 for i, _ in enumerate(new_counter_state)
                                           if old_counter_state[i] == "warm" or
                                           old_counter_state[i] == "cold"])
        num_food_delivered = sum([1 for i, _ in enumerate(new_counter_state)
                                  if (old_counter_state[i] == "warm" or old_counter_state[i] == "cold") and
                                  new_counter_state[i] == "empty"])

        num_food_burned = new_burned_total - old_burned_total
        num_food_not_burned = old_cooking_total - num_food_burned

        num_food_gone_cold = sum([1 for i, _ in enumerate(new_counter_state)
                                  if old_counter_state[i] == "warm" and new_counter_state[i] == "cold"])
        num_food_not_gone_cold = sum([1 for i, _ in enumerate(new_counter_state)
                                      if old_counter_state[i] == "warm" and new_counter_state[i] == "warm"])

        probability = burn_probability ** num_food_burned * \
                      (1 - burn_probability) ** num_food_not_burned * \
                      cold_probability ** num_food_gone_cold * \
                      (1 - cold_probability) ** num_food_not_gone_cold
        if num_food_delivered == 1:
            # only 1 dish can be delivered at a time so the probability of a dish being delivered is
            # the probability of any dish being delivered times the probability of that dish being the
            # one chosen for delivery
            probability *= deliver_probability * (1 / food_available_for_delivery)
        elif food_available_for_delivery:
            # if there is food available for delivery and nothing is delivered
            probability *= (1 - deliver_probability)
        return probability

    @staticmethod
    def get_stove_state_counts(stove_state):
        """
        :param stove_state: A tuple containing the state of all burners
        :return: A count of the total number of burners in each of the possible states
        """
        cooking_total = sum([1 for burner in stove_state if burner == "cooking"])
        burned_total = sum([1 for burner in stove_state if burner == "burned"])
        empty_burner_total = sum([1 for burner in stove_state if burner == "empty"])

        return cooking_total, burned_total, empty_burner_total

    @staticmethod
    def get_counter_state_counts(counter_state):
        warm_total = sum([1 for plate in counter_state if plate == "warm"])
        cold_total = sum([1 for plate in counter_state if plate == "cold"])
        empty_plate_total = sum([1 for plate in counter_state if plate == "empty"])

        return warm_total, cold_total, empty_plate_total

    def get_wait_states(self, state):
        """
        :param state: Input state to determine transitions from
        :return: A list of possible states that could result from the wait action being taken at state
        """
        # For every spot on the stove that is not cooking there is some probability of moving to burned
        # For every warm food on the counter there is some probability of moving to cold
        # Every dish on the counter has some probability of being delivered with priority going to cold food
        stove_state = state[0]
        counter_state = state[1]

        possible_stove_states = self.get_passive_stove_transitions(stove_state)
        possible_counter_states = self.get_passive_counter_transitions(counter_state)

        states = list(itertools.product(possible_stove_states, possible_counter_states))
        return states

    @staticmethod
    def get_passive_counter_transitions(counter_state):
        """
        :param counter_state: Tuple representing the state of the counter. Eg. ("empty", "warm", "warm", "cold")
        :return: All possible counter states that could occur after a transition where nothing is added to counter.
        In other words all combinations of warm food transitioning to cold and/or food transitioning to empty.
        """
        possible_new_counter_states = list()
        # Add all possible transitions from warm -> cold when food is NOT delivered
        possible_new_counter_states.extend(OvercookedEnvironment.get_possible_warm_to_cold_transitions(counter_state))

        # Add all possible transitions from warm -> cold when food is delivered
        # Compute the delivery state if there is a cold dish
        counter_states_after_delivery = list()
        if "cold" or "warm" in counter_state:
            for i, plate in enumerate(counter_state):
                if plate == "cold" or plate == "warm":
                    counter_state_after_delivery = list(counter_state)
                    counter_state_after_delivery[i] = "empty"
                    counter_states_after_delivery.append(tuple(counter_state_after_delivery))
        # Compute all possible transitions from warm -> cold for the counter if food was delivered
        for counter_state_after_delivery in counter_states_after_delivery:
            possible_new_counter_states.extend(
                OvercookedEnvironment.get_possible_warm_to_cold_transitions(counter_state_after_delivery))

        return possible_new_counter_states

    @staticmethod
    def get_possible_warm_to_cold_transitions(counter_state):
        """
        :param counter_state: A tuple containing the state of all plates on the counter
        :return: A list of tuples containing all possible new counter states that could occur if no action (wait) is
        taken from the passed in state.
        """
        warm_food_count = sum([1 for plate in counter_state if plate == "warm"])
        food_states = list(itertools.product(["warm", "cold"], repeat=warm_food_count))

        possible_new_counter_states = list()
        # Add all possible transitions from warm -> cold when food is NOT delivered
        for food_state in food_states:
            i = 0
            new_counter_state = list(counter_state)
            for j, plate in enumerate(new_counter_state):
                if plate == "warm":
                    new_counter_state[j] = food_state[i]
                    i += 1
            possible_new_counter_states.append(tuple(new_counter_state))

        return possible_new_counter_states

    @staticmethod
    def get_passive_stove_transitions(stove_state):
        """
        :param stove_state: Tuple representing the state of the stove. Eg. ("empty", "cooking", "cooking", "burned")
        :return: All possible stove states that could occur after a transition where nothing is added or removed
        from the stove. In other words all combinations of cooking transitioning to burned.
        """
        cooking_food_count = sum([1 for stove in stove_state if stove == "cooking"])
        cooking_states = list(itertools.product(["cooking", "burned"], repeat=cooking_food_count))

        possible_new_stove_states = list()
        for cooking_state in cooking_states:
            i = 0
            new_stove_state = list(stove_state)
            for j, burner in enumerate(new_stove_state):
                if burner == "cooking":
                    new_stove_state[j] = cooking_state[i]
                    i += 1
            possible_new_stove_states.append(tuple(new_stove_state))

        return possible_new_stove_states

    def check_trans(self, trans):
        # Check if the transitions are constructed correctly
        for st in trans.keys():
            for act in trans[st].keys():
                if abs(sum(trans[st][act].values()) - 1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(self.stotrans[st][act].values()))
                    return False
        print("Transition is correct")
        return True

    def chef_reward(self):
        # Reward function of the chef who is paid for food being cooked (moved to the counter from the stove)
        # Define the reward function of the MDP
        # Reward[state][action] = r
        reward = dict(zip(gw.states, [dict(zip(gw.actions, [0 for _ in gw.actions])) for _ in gw.states]))
        for state in self.states:
            stove_state = state[0]
            for action in self.actions:
                if "cooking" in stove_state and action == "move_food":
                    reward[state][action] = self.cook_food_reward
                if "burned" in stove_state:
                    reward[state][action] = self.burned_food_penalty
        return reward

    def leader_reward(self):
        # Reward function of the leader who provides the side payments to incentivize the chef
        reward = dict(zip(gw.states, [dict(zip(gw.actions, [0 for _ in gw.actions])) for _ in gw.states]))
        for state in self.states:
            counter_state = state[1]
            for action in self.actions:
                if "cold" in counter_state and action == "deliver":
                    reward[state][action] = self.cold_food_delivered_reward
                if "warm" in counter_state and action == "deliver":
                    reward[state][action] = self.warm_food_delivered_reward
                # Add reward based on probability of random delivery by waiter
                # R(s,a,s') -> R(s,a) = p * R(s,a,s')
                num_of_plates_with_food = sum([1 for plate in counter_state if plate == "warm" or plate == "cold"])
                for plate in counter_state:
                    if plate == "warm":
                        reward[state][action] += (self.deliver_probability / num_of_plates_with_food)\
                                                 * self.warm_food_delivered_reward
                    elif plate == "cold":
                        reward[state][action] += (self.deliver_probability / num_of_plates_with_food)\
                                                 * self.cold_food_delivered_reward
        return reward


def initial_MDP(stove_size, counter_size):
    GridW = OvercookedEnvironment(stove_size, counter_size)
    reward = GridW.reward_Goal_Obstacle()  # Initialize reward function
    pi = GridW.getpolicy_deter(reward)
    return GridW, pi


def main():
    stove_size = 3
    counter_size = 3
    GridWorld, Policy = initial_MDP(stove_size, counter_size)
    return GridWorld, Policy


if __name__ == "__main__":
    bp = 0.1
    cp = 0.2
    dp = 0.3
    gw = OvercookedEnvironment(2, 2, bp, cp, dp)
    print(gw.chef_reward()[(("empty", "empty"), ("empty", "empty"))])
    # for state in gw.transition:
    #     print(
    #         f"original_state: {state}\n transitions: {gw.transition[state]['add_food']}\n\n"
    #     )
    # for state in gw.transition:
    #     total_prob = sum([gw.transition[state]["move_food"][new_state]
    #                       for new_state in gw.transition[state]["move_food"]])
    #     if abs(total_prob-1) > 0.01:
    #         print(
    #             f"total_probability: {total_prob}\n initial_state: {state}\n transitions: {gw.transition[state]['move_food']}\n\n")
