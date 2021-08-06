import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from numpy import eye, hstack, ones, vstack, zeros
import sys

class soccer():

    def __init__(self, A, B):
        #players
        self.A = A
        self.B = B
        self.states = [0,1,2,3,4,5,6,7]
        self.starting = [1, 2, 5, 6]

        #goals
        self.goalA = [0,4]
        self.goalB = [3,7]

        #balls
        self.ball = None #0 if A and 1 if B
        self.ball_pos = None





    def start_game(self):
        #random
        '''
        #rand_start = np.random.choice(4,2) #choose two different numbers
        #self.A.pos = self.starting[rand_start[0]]
        #self.B.pos = self.starting[rand_start[1]]
        #choose who has the ball

        if np.random.random() < 0.5:
            self.ball = self.A.ball
            self.ball_pos = self.A.pos
        else:
            self.ball = self.B.ball
            self.ball_pos = self.B.pos
        '''
        #not random
        self.A.pos = 2
        self.B.pos = 1
        self.ball = 1
        self.ball = 0


    def move(self, player, action):
        #N,S,E,W,stay [0,1,2,3,4]
        act = ['N','S','E','W','O']
        dir = act[action]
        if dir == 'N' and player.pos > 3:
            pos = - 4
        elif dir == 'S' and player.pos < 4:
            pos = 4
        elif dir == 'E' and player.pos not in self.goalB:
            pos = 1
        elif dir == 'W' and player.pos not in self.goalA:
            pos = -1
        else: #stay
            pos = 0
        nextpos = player.pos + pos

        return nextpos

    def act(self, first, second, actfirst, actsecond):
        nextfirst = self.move(first, actfirst)
        nextsecond = self.move(second, actsecond)

        if nextfirst != second.pos: #can't move to B
            first.pos = nextfirst
        else: #if B is there, give the ball to B
            self.ball = second.ball

        if nextsecond != first.pos:
            second.pos = nextsecond
        else:
            self.ball = first.ball

        if self.ball:
            self.ball_pos = first.pos
        else:
            self.ball_pos = second.pos

    def next(self, actA, actB):
        rewA = 0
        rewB = 0
        finish = False
        #choose first player
        if np.random.random() < 0.5:
            self.act(self.A, self.B, actA, actB)
        else:
            self.act(self.B, self.A, actB, actA)
        #goal
        if self.ball_pos in self.goalA:
            rewA += 100
            rewB -= 100
            finish = True
        elif self.ball_pos in self.goalB:
            rewA -= 100
            rewB += 100
            finish = True
        return rewA, rewB, finish, self.cur()

    def cur(self):
        return self.A.pos, self.B.pos, self.ball, self.ball_pos



def maxmin(R):
    x = cp.Variable(5)
    r = np.array(R)
    exp = cp.sum(r @ x)
    contraints = [cp.sum(x) == 1, 0 <= x, r @ x == 0]
    objective = cp.Minimize(exp)
    problem = cp.Problem(objective, contraints)
    solution = problem.solve()

    return x.value

def system(X, A,B,n):
    row = 0
    for x in range(n):
        for y in range(n):
            if x != y:
                X[row, x * n:(x + 1) * n] = A[x] - A[y]
                X[row + n * (n - 1), x:(n * n):n] = B[:, x] - B[:, y]
                row += 1
    return X, A, B

def cvxopt_maximin(q):

    R = matrix(q).trans()
    n = R.size[1] #length of col
    X = hstack((ones((R.size[0], 1)), R)) #stack horizontally
    R2 = hstack((zeros((n, 1)), -eye(n))) #make matrix > 0
    X = vstack((X, R2)) #stack vertically
    X = matrix(vstack((X, hstack((0, ones(n))), hstack((0, -ones(n)))))) #make sum == 1
    Y = matrix(hstack((zeros(X.size[0] - 2), [1, -1])))
    Z = matrix(hstack(([-1], zeros(n))))
    glpksolver = 'glpk'
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    solvers.options['msg_lev'] = 'GLP_MSG_OFF'
    solvers.options['LPX_K_MSGLEV'] = 0
    solution = solvers.lp(Z, X, Y, solver=glpksolver)

    return solution['primal objective']

'''
def ceq_maxmin(Q_A, Q_B):
    qA = np.array(Q_A)
    qB = np.array(Q_B)

    x = cp.Variable(5,5)
    y = cp.Variable(1)
    pSum = cp.sum(x)
    constraints = [pSum >= 0, pSum == 1]
    qSum = cp.sum(qA @ qB)
    constraints.append((qSum == y))
'''


def ceq(A, B):

    R = matrix(A).trans()
    n = R.size[1]
    X = np.zeros((2 * n * (n - 1), (n * n))) #make canvas 40 * 25
    A = np.array(A) #5x5
    B = np.array(B) #5x5

    #make equation
    X, A, B = system(X, A, B, n)

    X = matrix(X) #make cvxopt matrix
    X = hstack((ones((X.size[0], 1)), X))
    R2 = hstack((zeros((n * n, 1)), -eye(n * n))) #make > 0
    X = vstack((X, R2))
    X = matrix(vstack((X, hstack((0, ones(n * n))), hstack((0, -ones(n * n)))))) #make sum == 1
    Y = matrix(hstack((zeros(X.size[0] - 2), [1, -1])))
    Z = matrix(hstack(([-1.], -(A + B).flatten()))) #make sums flat
    glpksolver = 'glpk'
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    solvers.options['msg_lev'] = 'GLP_MSG_OFF'
    solvers.options['LPX_K_MSGLEV'] = 0
    solution = solvers.lp(Z, X, Y, solver=glpksolver)

    # CE_Q
    if solution['x'] is None: #if solution doesn't exist
        return 0, 0
    dist = solution['x'][1:] #prob dist
    A2 = A.flatten()
    B2 = B.transpose().flatten()

    #get the expected values
    A_exp = np.matmul(A2, dist)[0]
    B_exp = np.matmul(B2, dist)[0]

    return A_exp, B_exp


class player():

    def __init__(self, ball): #0 for A and 1 for B
        self.rewards = 0
        self.pos = None
        self.ball = ball

class functions():

    def __init__(self):
        pass

    def Q_learning(self, env, A, B):
        iter = 1000000
        epsilon = 0.9
        #epsilon_decay = 0.89/iter
        epsilon_min = 0.01
        alpha = 1
        alpha_min = 0.001
        alpha_decay =0.999/iter
        gamma = 0.9
        exp_q = [2, 1, 1, 1, 4]
        Q_A = np.zeros([8, 8, 2, 5]) #Apos * Bpos * ball * actions
        Q_B = np.zeros([8, 8, 2, 5])
        error = []
        x = []
        env.start_game()
        finish = False
        #for every iteration
        for i in range(iter):
            #restart if game is finished
            if finish:
                env.start_game()
                finish = False
            a_pos = A.pos
            b_pos = B.pos
            ball = env.ball
            exp_q_val = Q_A[2, 1, 1, 1]
            #choose random action based on epsilon
            if epsilon < np.random.random():
                actA = np.random.randint(5)
                actB = np.random.randint(5)
            else:
                actA = np.argmax(Q_A[a_pos,b_pos,ball])
                actB = np.argmax(Q_B[a_pos,b_pos,ball])

            #find next steps
            rewA, rewB, finish, next = env.next(actA,actB)
            nextA, nextB, nextball, ballpos = next

            #Q-learning
            Q_A[a_pos,b_pos,ball,actA] = (1 - alpha) * Q_A[a_pos ,b_pos ,ball, actA] + alpha * ((1 - gamma) * rewA + gamma * np.max(Q_A[nextA, nextB, nextball]))

            #error cal, Q-value difference
            if exp_q == [a_pos, b_pos, ball, actA, actB]:
                error.append(abs(Q_A[2, 1, 1, 1] - exp_q_val))
                x.append(i)
                #print("Iteration: ", i, alpha)
            #alpha decay
            if alpha > alpha_min:
                alpha -= alpha_decay
                #alpha *= 0.999995

            #epsilon decay
            if epsilon > epsilon_min:
                epsilon *= 0.9999
                #epsilon -= epsilon_decay
            if i % 1000 == 0:
                print(i)
        return error, x



    def Friend_Q(self, env, A, B):

        iter = 1000000
        alpha = 0.2
        n_sa = 0
        alpha_min = 0.001
        #alpha_decay = 0.499/iter
        gamma = 0.9
        exp_q = [2, 1, 1, 1, 4] #state in the paper
        Q_A = np.zeros([8, 8, 2, 5, 5]) #Apos, Bpos, ball, actA, actB
        error = []
        x = []
        env.start_game()
        finish = False

        for i in range(iter):

            if finish:
                env.start_game()
                finish = False
            a = A.pos
            b = B.pos
            ball = env.ball
            exp_q_val = Q_A[2, 1, 1, 1, 4] #get Q-value for the state in the paper
            #just choose random action, non-deterministic
            actA = np.random.randint(5)
            actB = np.random.randint(5)

            #next step
            rewA, rewB, finish, next = env.next(actA, actB)
            nextA, nextB, nextball, ballpos = next

            #Q-learning
            Q_A[a, b, ball, actA, actB] = (1 - alpha) * Q_A[a, b, ball, actA, actB] + alpha * ((1 - gamma) * rewA + gamma * np.max(Q_A[nextA, nextB, nextball]))

            #error, Q-value difference
            if exp_q == [a, b, ball, actA, actB]:
                error.append(abs(Q_A[2, 1, 1, 1, 4] - exp_q_val))
                x.append(i)

            #alpha decay
            if alpha > alpha_min:
                alpha *= 0.99995
                #alpha = 1 / n_sa
                #alpha -= alpha_decay

            if i % 1000 == 0:
                print(i)
        return error, x


    def FOE_Q(self, env, A, B):

        iter = 1000000
        alpha = 0.9
        alpha_min = 0.001
        alpha_decay = 0.9999
        gamma = 0.9
        exp_q = [2 ,1, 1, 1, 4]
        Q = np.zeros([8,8,2,5,5]) #Apos, Bpos, ball, actA, actB
        error = []
        x = []
        env.start_game()
        done = False
        np.random.seed(0)
        for i in range(iter):

            if done:
                env.start_game()
                done = False

            a = A.pos
            b = B.pos
            ball = env.ball
            exp_q_val = Q[2, 1, 1, 1, 4]
            cur = Q[a, b, ball]
            actA = np.random.randint(5)
            actB = np.random.randint(5)

            rewA, rewB, done, next = env.next(actA, actB)
            nextA, nextB, nextball, ballpos = next

            nextQ = Q[nextA, nextB, nextball]

            solve = cvxopt_maximin(cur) #linear programming, maxmin

            #Q-learning
            Q[a, b, ball, actA, actB] = (1 - alpha) * Q[a, b, ball, actA, actB] + alpha * ((1 - gamma) * rewA + gamma * solve)

            #error, Q-value difference
            if exp_q == [a, b, ball, actA, actB]:
                error.append(abs(Q[2, 1, 1, 1, 4] - exp_q_val))
                x.append(i)

            # alpha = 1/n_sa
            #n_sa += 1
            if alpha > alpha_min:
                alpha *= 0.99999

            if i % 100 == 0:
                print(i)
        return error, x


    def CE_Q(self, env, A, B):

        iter = 1000000
        alpha = 1
        alpha_min = 0.001
        #alpha_decay = 0.999/iter
        gamma = 0.9
        exp_q = [2, 1, 1, 1, 4]
        Q_A = np.zeros([8,8,2,5,5])  # states * states * actions
        Q_B = np.zeros([8,8,2,5,5])
        error = []
        x = []
        env.start_game()
        done = False
        np.random.seed(0)
        for i in range(iter):

            if done:
                env.start_game()
                done = False

            a = A.pos
            b = B.pos
            ball = env.ball
            exp_q_val = Q_A[2, 1, 1, 1, 4]
            actA = np.random.randint(5)
            actB = np.random.randint(5)

            cur_A = Q_A[a, b, ball]
            cur_B = Q_B[a, b, ball]

            rewA, rewB, done, next = env.next(actA, actB)
            nextA, nextB, nextball, ballpos = next

            corrA, corrB = ceq(cur_A, cur_B)

            Q_A[a, b, ball, actA] = (1 - alpha) * Q_A[a, b, ball, actA] + alpha * ((1 - gamma) * rewA + gamma * corrA)
            Q_B[a, b, ball, actB] = (1 - alpha) * Q_B[a, b, ball, actA] + alpha * ((1 - gamma) * rewB + gamma * corrB)

            if exp_q == [a, b, ball, actA, actB]:
                error.append(abs(Q_A[2, 1, 1, 1, 4] - exp_q_val))
                x.append(i)

            # alpha = 1/n_sa
            # n_sa += 1
            if alpha > alpha_min:
                alpha *= 0.99999
            '''
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                # epsilon -= epsilon_decay
            '''
            if i % 1000 == 0:
                print(i)
        return error, x

def graph(error, x, title, y):
    plt.axis([0,10 ** 6, 0, y])
    plt.plot(x, error)
    plt.xlabel('Iterations')
    plt.ylabel('Q-Value Difference')
    plt.title(title)
    plt.show()
    plt.gcf().clear()

if __name__ == "__main__":

    A = player(0)
    B = player(1)
    soccer = soccer(A,B)
    function = functions()
    if sys.argv[1] == 'Q-learning':
        q, x1 = function.Q_learning(soccer, A, B)
        graph(q, x1, "Q-Learning", 0.5)
    elif sys.argv[1] == 'Friend-Q':
        friend, x2 = function.Friend_Q( soccer, A, B)
        graph(friend, x2, "Friend-Q", 0.5)
    elif sys.argv[1] == 'Foe-Q':
        foe, x3 = function.FOE_Q(soccer, A, B)
        graph(foe, x3,"FOE_Q", 0.5)
    elif sys.argv[1] == 'CE-Q':
        ce, x4 = function.CE_Q(soccer, A, B)
        graph(ce, x4,"CE_Q",0.5)
    else:
        print("ex: python soccer.py Q-learning")