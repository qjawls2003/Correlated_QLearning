import copy
import random
from collections import defaultdict

from cvxopt.modeling import op
from cvxopt.modeling import variable
from cvxopt.solvers import options

from soccer.actions import Actions
from soccer.player import Player
from soccer.soccer_game import SoccerGame
from soccer.state import State


class Solver:
    def __init__(self):
        # Actions players can take
        self.actions = [Actions.UP, Actions.DOWN, Actions.RIGHT, Actions.LEFT,
                        Actions.STICK]

        # Starting game state for the soccer match
        self.init_state = State(Player(1, (3, 0), False),
                                Player(2, (1, 0), True))

        # State for gathering Q-value differences
        self.q_stat_state = State(Player(1, (2, 0), False),
                                  Player(2, (1, 0), True))

        # V tables for players A and B initialized to 1
        self.V1 = defaultdict(lambda: 1)
        self.V2 = defaultdict(lambda: 1)

    def q_learning(self, time_steps, alpha, gamma):
        """
        Normal Q-learning algorithm used in MDP's, completely ignores the
        other player.
        :param time_steps: Number of time steps to train on.
        :param alpha: Learning rate
        :param gamma: Discount factor
        :return: Statistics gathered during training
        """
        # List of gathered statistics
        statistics = list()

        # State action pair to gather q-value differences from
        q_stat = (self.q_stat_state, Actions.DOWN)

        # Q-table
        Q = defaultdict(lambda: 1)

        # Time step counter
        time_step_counter = 0

        # Soccer game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # Restart game if ended already.
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # Current state
            cur_state = copy.deepcopy(game.state)

            # Select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # Apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log time step and alpha
            if time_step_counter % 10000 == 0:
                print(time_step_counter)
                print(alpha)

            # Get reward
            current_reward = game.state.reward_value()

            # If reward not zero, game is now over.
            if current_reward != 0:
                game_over = True

            # Set value of new state
            self.V1[game.state] = max(Q[(game.state, Actions.UP)],
                                      Q[(game.state, Actions.DOWN)],
                                      Q[(game.state, Actions.RIGHT)],
                                      Q[(game.state, Actions.LEFT)],
                                      Q[(game.state, Actions.STICK)])

            # previous q-value
            pre_q = Q[(cur_state, a)]

            # q-update
            Q[(cur_state, a)] = (1 - alpha) * Q[(cur_state, a)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post update q-value
            post_q = Q[(cur_state, a)]

            # record stats if in correct state action pair
            if (cur_state, a) == q_stat:
                statistics.append(
                    (time_step_counter, abs(post_q - pre_q), pre_q, post_q))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format is (time-step, q-diff, pre-q-val, post-q-val)
        return statistics

    def friend_q_learning(self, time_steps, alpha, gamma):
        """
        Friend Q-Learning algorithm. Assumes the other player is an ally and
        will always attempt to help.
        :param time_steps: Number of time steps to train on.
        :param alpha: Learning rate
        :param gamma: Discount factor
        :return: Statistics gathered during training.
        """
        # gathered stats
        statistics = list()

        # state joint action pair to record q-diff's
        q_stat = (self.q_stat_state, Actions.DOWN, Actions.STICK)

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start soccer game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # take actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log time-step and alpha
            if time_step_counter % 10000 == 0:
                print(time_step_counter)
                print(alpha)

            # get current reward
            current_reward = game.state.reward_value()

            # if not 0, game is over
            if current_reward != 0:
                game_over = True

            # get max q-value
            max_q_value = Q[(game.state, Actions.UP, Actions.UP)]
            for p1_a in self.actions:
                for p2_o in self.actions:
                    max_q_value = max(max_q_value, Q[(game.state, p1_a, p2_o)])

            # update value of state
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # q-value update
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # record q-diff's
            if (cur_state, a, o) == q_stat:
                statistics.append(
                    (time_step_counter, abs(post_q - pre_q), pre_q, post_q))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val)
        return statistics

    def foe_q_learning(self, time_steps, alpha, gamma):
        """
        Foe Q-learning. Assumes the other player will always attempt to minimize
        its reward.
        :param time_steps: Number of time steps to train on.
        :param alpha: Learning rate
        :param gamma: Discount factor
        :return: Statistics gathered during training.
        """
        # turn lp solver logging off
        options['show_progress'] = False

        # gathered stats
        statistics = list()

        # state joint action pair to record q-diffs in
        q_stat = (self.q_stat_state, Actions.DOWN, Actions.STICK)

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log info
            if time_step_counter % 10000 == 0:
                print(time_step_counter)
                print(alpha)

            # get current reward
            current_reward = game.state.reward_value()
            if current_reward != 0:
                game_over = True

            # action probabilities
            probs = list()
            for i in range(len(self.actions)):
                probs.append(variable())

            # all action probabilities >= 0 constraints
            constrs = list()
            for i in range(len(self.actions)):
                constrs.append((probs[i] >= 0))

            # sum of probabilities = 1 constraint
            total_prob = sum(probs)
            constrs.append((total_prob == 1))

            # objective
            v = variable()

            # set mini-max constraints
            for j in range(5):
                c = 0
                for i in range(5):
                    c += Q[(game.state, self.actions[i], self.actions[j])] * \
                         probs[i]
                constrs.append((c >= v))

            # maximize objective
            lp = op(-v, constrs)
            lp.solve()

            # set value of state
            max_q_value = v.value[0]
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # update q-value
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # gather statistics
            if (cur_state, a, o) == q_stat:
                prob_list = list()
                for i in range(len(probs)):
                    prob_list.append(probs[i].value[0])
                statistics.append((
                                  time_step_counter, abs(post_q - pre_q), pre_q,
                                  post_q, prob_list))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val, probabilities)
        return statistics

    def ce_q_learning(self, time_steps, alpha, gamma):
        """
        Correlated Q-Learning algorithm. Players share their knowledge of the
        game, and attempt to maximize some common goal. In this case, they are
        attempting to maximize the sum of their rewards.
        :param time_steps: Number of time steps to train on.
        :param alpha: Learning rate
        :param gamma: Discount factor
        :return: Statistics gathered during training
        """
        # Turn off lp logging
        options['show_progress'] = False

        # gathered stats
        statistics = list()
        q_stat = (self.q_stat_state, Actions.DOWN, Actions.STICK)

        # Q-tables for each player
        Q1 = defaultdict(lambda: 1)
        Q2 = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # start game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # choose actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log info
            if time_step_counter % 1000 == 0:
                print(time_step_counter)
                print(alpha)

            # get current reward
            current_reward = game.state.reward_value()
            if current_reward != 0:
                game_over = True

            # joint action probabilities
            probs = {}
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    probs[(i, j)] = variable()

            # all joint action probabilities >= 0 constraints
            constrs = list()
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    constrs.append((probs[(i, j)] >= 0))

            # sum of joint action probabilities = 1
            total_prob = 0
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    total_prob += probs[(i, j)]
            constrs.append((total_prob == 1))

            # objective
            v = variable()

            # 20 rationality constraints for player A
            for i in range(len(self.actions)):
                rc1 = 0
                for j in range(len(self.actions)):
                    rc1 += probs[(i, j)] * Q1[
                        (game.state, self.actions[i], self.actions[j])]

                for k in range(len(self.actions)):
                    if i != k:
                        rc2 = 0
                        for l in range(len(self.actions)):
                            rc2 += probs[(i, l)] * Q1[
                                (game.state, self.actions[k], self.actions[l])]
                        constrs.append((rc1 >= rc2))

            # 20 rationality constraints for player B
            for i in range(len(self.actions)):
                rc1 = 0
                for j in range(len(self.actions)):
                    rc1 += probs[(j, i)] * Q2[
                        (game.state, self.actions[j], self.actions[i])]

                for k in range(len(self.actions)):
                    if i != k:
                        rc2 = 0
                        for l in range(len(self.actions)):
                            rc2 += probs[(l, i)] * Q2[
                                (game.state, self.actions[l], self.actions[k])]
                        constrs.append((rc1 >= rc2))

            # sum of the players rewards
            sum_total = 0
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    sum_total += probs[(i, j)] * Q1[
                        (game.state, self.actions[i], self.actions[j])]
                    sum_total += probs[(i, j)] * Q2[
                        (game.state, self.actions[i], self.actions[j])]

            constrs.append((v == sum_total))

            # maximize sum of players rewards
            lp = op(-v, constrs)
            lp.solve()

            if lp.status == 'optimal':
                # update V table for player A
                v1 = 0
                for i in range(len(self.actions)):
                    for j in range(len(self.actions)):
                        v1 += probs[(i, j)].value[0] * Q1[
                            (game.state, self.actions[i], self.actions[j])]
                self.V1[game.state] = v1

                # update V table for player B
                v2 = 0
                for i in range(len(self.actions)):
                    for j in range(len(self.actions)):
                        v2 += probs[(i, j)].value[0] * Q2[
                            (game.state, self.actions[i], self.actions[j])]
                self.V2[game.state] = v2

            # pre q-value
            pre_q = Q1[(cur_state, a, o)]

            # P1 Q update
            Q1[(cur_state, a, o)] = (1 - alpha) * Q1[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # P2 Q Update
            Q2[(cur_state, a, o)] = (1 - alpha) * Q2[
                (cur_state, a, o)] + alpha * (
                -current_reward + gamma * self.V2[game.state])

            # post q-value
            post_q = Q1[(cur_state, a, o)]

            # gather stats
            if (cur_state, a, o) == q_stat:
                prob_list = list()
                for x in range(5):
                    for y in range(5):
                        prob_list.append(probs[(x, y)].value[0])
                statistics.append((
                                  time_step_counter, abs(post_q - pre_q), pre_q,
                                  post_q, prob_list))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val, probabilities)
        return statistics

from enum import Enum


class Actions(Enum):
    """
    Enum of actions available to the players during the soccer match
    """
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    STICK = 5

class Player:
    """
    Soccer player
    """

    def __init__(self, num, cords, has_ball):
        """
        :param num: player number, should be unique
        :param cords: coordinates of player in the soccer field. Two players
        should not occupy the same space.
        :param has_ball: boolean, True if player has possession of the ball.
        """
        self.num = num
        self.cords = cords
        self.has_ball = has_ball

    def __eq__(self, other):
        return self.num == other.num and self.cords == other.cords \
               and self.has_ball == other.has_ball

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.num, self.cords, self.has_ball))

    def __str__(self):
        return str((self.num, self.cords, self.has_ball))

import copy
import random


class SoccerGame:
    """
    Soccer game. Intended only for a 2 player match.
    """

    def __init__(self, state):
        """
        :param state: State the game should start in.
        """
        self.state = copy.deepcopy(state)

    def apply_actions(self, a, o):
        """
        Both players choose actions simultaneously, however actions are not
        executed simultaneously. There is a 50% chance player 1 will go before
        player 2.
        :param a: Action player 1 is attempting.
        :param o: Action player 2 is attempting.
        :return: None
        """
        self.state = random.sample(self.state.get_reachable_states(a, o), 1)[0]

from soccer.actions import Actions
from soccer.player import Player


class State:
    # Dimensions of grid soccer field.
    FIELD_DIMENSIONS = [4, 2]

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def get_reachable_states(self, a, o):
        """
        :param a: player 1's action
        :param o: player 2's action
        :return: Set of possible states after actions are executed.
        """
        reachable_states = set()

        # If player A moves first
        new_cords_a = self.new_player_cords(self.player1, a)
        new_cords_b = self.new_player_cords(self.player2, o)

        if new_cords_a == self.player2.cords:
            if self.player1.has_ball:
                tmp_a = Player(1, self.player1.cords, False)
                tmp_b = Player(2, self.player2.cords, True)
            else:
                tmp_a = Player(1, self.player1.cords, False)
                tmp_b = Player(2, self.player2.cords, True)
        else:
            if self.player1.has_ball:
                tmp_a = Player(1, new_cords_a, True)
                if new_cords_a == new_cords_b:
                    tmp_b = Player(2, self.player2.cords, False)
                else:
                    tmp_b = Player(2, new_cords_b, False)
            else:
                if new_cords_a == new_cords_b:
                    tmp_a = Player(1, new_cords_a, True)
                    tmp_b = Player(2, self.player2.cords, False)
                else:
                    tmp_a = Player(1, new_cords_a, False)
                    tmp_b = Player(2, new_cords_b, True)

        reachable_states.add(State(tmp_a, tmp_b))

        # If player B moves first
        new_cords_a = self.new_player_cords(self.player1, a)
        new_cords_b = self.new_player_cords(self.player2, o)
        if new_cords_b == self.player1.cords:
            if self.player2.has_ball:
                tmp_b = Player(2, self.player2.cords, False)
                tmp_a = Player(1, self.player1.cords, True)
            else:
                tmp_b = Player(2, self.player2.cords, False)
                tmp_a = Player(1, self.player1.cords, True)
        else:
            if self.player2.has_ball:
                tmp_b = Player(2, new_cords_b, True)
                if new_cords_b == new_cords_a:
                    tmp_a = Player(1, self.player1.cords, False)
                else:
                    tmp_a = Player(1, new_cords_a, False)
            else:
                if new_cords_b == new_cords_a:
                    tmp_b = Player(2, new_cords_b, True)
                    tmp_a = Player(1, self.player1.cords, False)
                else:
                    tmp_b = Player(2, new_cords_b, False)
                    tmp_a = Player(1, new_cords_a, True)

        reachable_states.add(State(tmp_a, tmp_b))

        return reachable_states

    def __eq__(self, other):
        return self.player1 == other.player1 and self.player2 == other.player2

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.player1, self.player2))

    def __str__(self):
        return str((str(self.player1), str(self.player2)))

    def reward_value(self):
        """
        :return: The reward for player 1, since its a zero sum game, player 2's
        reward is the negation of the reward.
        """
        if self.player1.has_ball:
            x, y = self.player1.cords
            if x == 0:
                return 100
            elif x == State.FIELD_DIMENSIONS[0] - 1:
                return -100
            else:
                return 0
        elif self.player2.has_ball:
            x, y = self.player2.cords
            if x == 0:
                return 100
            elif x == State.FIELD_DIMENSIONS[0] - 1:
                return -100
            else:
                return 0

    @staticmethod
    def new_player_cords(player, action):
        """
        :param player: Some player
        :param action: Action player has selected to take
        :return: tuple of coordinates where new player will be if action is
        successful.
        """
        x, y = player.cords
        if action == Actions.UP:
            y = max(0, y - 1)
        elif action == Actions.DOWN:
            y = min(State.FIELD_DIMENSIONS[1] - 1, y + 1)
        elif action == Actions.LEFT:
            x = max(0, x - 1)
        elif action == Actions.RIGHT:
            x = min(State.FIELD_DIMENSIONS[0] - 1, x + 1)
        return x, y