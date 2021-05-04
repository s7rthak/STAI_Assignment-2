import matplotlib.pyplot as plt
import numpy as np
import matplotlib.markers as markers

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = set()
        self.goal = None

class Agent:
    def __init__(self, x, y, grid):
        self.x = x
        self.y = y
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.grid = grid
        self.state_history = [(x, y)]
        self.nominal_action_history = []
        self.actual_action_history = []
        self.reward_history = []
        self.reached_goal = False

    def execute_actual_action(self, action):
        x_dash, y_dash = None, None
        if action == 'Up':
            x_dash, y_dash = self.x, self.y + 1
        elif action == 'Down':
            x_dash, y_dash = self.x, self.y - 1
        elif action == 'Left':
            x_dash, y_dash = self.x - 1, self.y
        elif action == 'Right':
            x_dash, y_dash = self.x + 1, self.y
        
        if (self.x, self.y) == self.grid.goal:
            self.reward_history.append(0)
            self.state_history.append((self.x, self.y))
        elif (x_dash, y_dash) in self.grid.walls:
            self.reward_history.append(-1)
            self.state_history.append((self.x, self.y))
        elif (x_dash, y_dash) == self.grid.goal:
            self.reward_history.append(100)
            self.state_history.append((x_dash, y_dash))
            self.x = x_dash
            self.y = y_dash
        else:
            self.reward_history.append(0)
            self.state_history.append((x_dash, y_dash))
            self.x = x_dash
            self.y = y_dash

    def execute_nominal_action(self, action):
        self.nominal_action_history.append(action)
        if action == 'Up':
            actual_action = np.random.choice(self.actions, 1, p = [0.8, 0.2/3, 0.2/3, 0.2/3])[0]
        if action == 'Down':
            actual_action = np.random.choice(self.actions, 1, p = [0.2/3, 0.8, 0.2/3, 0.2/3])[0]
        if action == 'Left':
            actual_action = np.random.choice(self.actions, 1, p = [0.2/3, 0.2/3, 0.8, 0.2/3])[0]
        if action == 'Right':
            actual_action = np.random.choice(self.actions, 1, p = [0.2/3, 0.2/3, 0.2/3, 0.8])[0]
        self.execute_actual_action(actual_action)

    def clear_history(self, x, y):
        self.x = x
        self.y = y
        self.state_history = [(x, y)]
        self.nominal_action_history = []
        self.actual_action_history = []
        self.reward_history = []

def choose_eps_greedy_action(Q, agent, S, eps):
    all_q_values = [Q[(S, a)] for a in agent.actions]
    max_q_value = max(all_q_values)
    pos_actions = [a for a in agent.actions if Q[(S, a)] == max_q_value]
    best_action = np.random.choice(pos_actions, 1)[0]
    weights = [eps/len(agent.actions) if a != best_action else 1 - eps + eps/len(agent.actions) for a in agent.actions]

    return np.random.choice(agent.actions, 1, p = weights)[0]

eps = 0.005
for i in range(3):
    # Simulation begins here.
    grid_world = Grid(50, 25)
    grid_world.goal = (48, 12)          # defining the goal-state.

    # Adding walls to grid.
    for i in range(50):
        grid_world.walls.add((i, 0))
        grid_world.walls.add((i, 24))
    for i in range(1, 24):
        grid_world.walls.add((0, i))
        grid_world.walls.add((49, i))
    for i in range(1, 12):
        grid_world.walls.add((25, i))
        grid_world.walls.add((26, i))
    for i in range(13, 24):
        grid_world.walls.add((25, i))
        grid_world.walls.add((26, i))

    # Random start
    def random_start ():
        x, y = np.random.randint(0, 50), np.random.randint(0, 25)
        while (x, y) in grid_world.walls:
            x, y = np.random.randint(0, 50), np.random.randint(0, 25)
        return x, y

    Q = dict()

    for i in range(50):
        for j in range(25):
            if (i, j) not in grid_world.walls:
                for action in ['Up', 'Down', 'Left', 'Right']:
                    Q[((i, j), action)] = 0

    num_episodes = 4000
    steps_in_episodes = 1000
    alpha = 0.25
    discount = 0.99
    mobile_agent = Agent(1, 1, grid_world)

    for i in range(num_episodes):
        mobile_agent.clear_history(*random_start())
        S = (mobile_agent.x, mobile_agent.y)
        for j in range(steps_in_episodes):
            A = choose_eps_greedy_action(Q, mobile_agent, S, eps)
            mobile_agent.execute_nominal_action(A)
            S_dash, R = mobile_agent.state_history[-1], mobile_agent.reward_history[-1]
            Q[(S, A)] = Q[(S, A)] + alpha * (R + discount * max([Q[(S_dash, a)] for a in mobile_agent.actions]) - Q[(S, A)])
            S = S_dash

            if S == grid_world.goal:
                # print("broke :(")
                break

    Pi = [[None for j in range(25)] for i in range(50)]
    V = np.zeros((50, 25))

    for i in range(50):
        for j in range(25):
            if (i, j) not in grid_world.walls:
                best_value = max([Q[((i, j), a)] for a in mobile_agent.actions])
                pos_policy = [(Q[((i, j), a)], a) for a in mobile_agent.actions if Q[((i, j), a)] == best_value]
                (value, action) = pos_policy[np.random.choice(len(pos_policy), 1)[0]]
                V[i, j] = value
                Pi[i][j] = action

    fig = plt.figure(figsize=(16, 8))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 51, 1))
    ax.set_yticks(np.arange(0, 26, 1))
    ax.set_xlim([-0.5, 49.5])
    ax.set_ylim([-0.5, 24.5])
    marker = markers.MarkerStyle(marker='s')
    P = np.arange(25)
    Q = np.arange(50)
    all_points = np.dstack(np.meshgrid(P, Q)).reshape(-1, 2)
    X = V.reshape(-1, 1)
    all_walls = list(grid_world.walls)
    wall_color = ['r']*len(all_walls)
    # print(X)
    # print(all_points)
    scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
    scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)

    def arrow_coordinates(i, j, action):
        if action == 'Up':
            return i, j + 0.1, 0, 0.7
        elif action == 'Down':
            return i, j - 0.1, 0, -0.7
        elif action == 'Left':
            return i - 0.1, j, -0.7, 0
        elif action == 'Right':
            return i + 0.1, j, 0.7, 0

    all_arrows = []
    for i in range(50):
        for j in range(25):
            if Pi[i][j] != None:
                x, y, dx, dy = arrow_coordinates(i, j, Pi[i][j])
                all_arrows.append((x, y, dx, dy))

    for i in range(len(all_arrows)):
        plt.arrow(all_arrows[i][0], all_arrows[i][1], all_arrows[i][2], all_arrows[i][3], length_includes_head=True, head_width=0.15, edgecolor='royalblue', facecolor='y')

    plt.savefig('c_eps_' + str(eps)+'.png', bbox_inches='tight')
    plt.show()
    eps = eps * 10
