import matplotlib.pyplot as plt
import numpy as np
import matplotlib.markers as markers
import matplotlib.colors as colors

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
        
        if (x_dash, y_dash) in self.grid.walls:
            if (self.x, self.y) == self.grid.goal:
                self.reward_history.append(100)
            else:
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
        self.actual_action_history.append(actual_action)
        self.execute_actual_action(actual_action)

    def clear_history(self, x, y):
        self.x = x
        self.y = y
        self.state_history = [(x, y)]
        self.nominal_action_history = []
        self.actual_action_history = []
        self.reward_history = []

def value_update(V, V_dash, i, j, grid, discount):
    if (i, j) in grid.walls:
        return 0
    all_actions = ['Up', 'Down', 'Left', 'Right']
    T = np.zeros((5, 4))
    R = np.zeros((5, 4))
    x_up, y_up = i, j + 1
    x_down, y_down = i, j - 1
    x_left, y_left = i - 1, j
    x_right, y_right = i + 1, j
    
    values = []
    for action in all_actions:
        tmp = 0.0
        if action == 'Up':
            if (x_up, y_up) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[i, j])
                else:
                    tmp += 0.8 * (-1 + discount * V[i, j])
            else:
                if (x_up, y_up) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[x_up, y_up])
                else:
                    tmp += 0.8 * (0 + discount * V[x_up, y_up])

            if (x_down, y_down) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_down, y_down) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_down, y_down])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_down, y_down])

            if (x_left, y_left) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_left, y_left) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_left, y_left])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_left, y_left])

            if (x_right, y_right) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_right, y_right) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_right, y_right])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_right, y_right])
        
        elif action == 'Down':
            if (x_up, y_up) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_up, y_up) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_up, y_up])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_up, y_up])

            if (x_down, y_down) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[i, j])
                else:
                    tmp += 0.8 * (-1 + discount * V[i, j])
            else:
                if (x_down, y_down) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[x_down, y_down])
                else:
                    tmp += 0.8 * (0 + discount * V[x_down, y_down])

            if (x_left, y_left) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_left, y_left) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_left, y_left])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_left, y_left])

            if (x_right, y_right) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_right, y_right) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_right, y_right])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_right, y_right])

        elif action == 'Left':
            if (x_up, y_up) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_up, y_up) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_up, y_up])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_up, y_up])

            if (x_down, y_down) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_down, y_down) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_down, y_down])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_down, y_down])

            if (x_left, y_left) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[i, j])
                else:
                    tmp += 0.8 * (-1 + discount * V[i, j])
            else:
                if (x_left, y_left) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[x_left, y_left])
                else:
                    tmp += 0.8 * (0 + discount * V[x_left, y_left])

            if (x_right, y_right) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_right, y_right) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_right, y_right])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_right, y_right])

        elif action == 'Right':
            if (x_up, y_up) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_up, y_up) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_up, y_up])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_up, y_up])

            if (x_down, y_down) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_down, y_down) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_down, y_down])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_down, y_down])

            if (x_left, y_left) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[i, j])
                else:
                    tmp += 0.2/3 * (-1 + discount * V[i, j])
            else:
                if (x_left, y_left) == grid.goal:
                    tmp += 0.2/3 * (100 + discount * V[x_left, y_left])
                else:
                    tmp += 0.2/3 * (0 + discount * V[x_left, y_left])

            if (x_right, y_right) in grid.walls:
                if (i, j) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[i, j])
                else:
                    tmp += 0.8 * (-1 + discount * V[i, j])
            else:
                if (x_right, y_right) == grid.goal:
                    tmp += 0.8 * (100 + discount * V[x_right, y_right])
                else:
                    tmp += 0.8 * (0 + discount * V[x_right, y_right])
        values.append(tmp)
    
    V_dash[i, j] = max(values)
    return abs(V_dash[i, j] - V[i, j])


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

# x, y = None, None
# while (x, y) in grid_world.walls:
#     x, y = np.random.randint(0, 50), np.random.randint(0, 25)

# mobile_agent = Agent(x, y, grid_world)

# Take a random policy.
# arbitrary_policy = []
# for i in range(50):
#     tmp = []
#     for j in range(25):
#         tmp.append(np.random.choice(mobile_agent.actions, 1)[0])
#     arbitrary_policy.append(tmp)

V = np.zeros((50, 25))              # initialize values

iterations = 0
delta = float("inf")
theta = 0.1
discount = 0.99

while iterations < 100 and delta > theta:
    V_dash = np.zeros((50, 25))
    diff = []
    for i in range(50):
        for j in range(25):
            diff.append(value_update(V, V_dash, i, j, grid_world, discount))
    delta = max(diff)
    V = V_dash
    iterations += 1

if delta > theta:
    print("Completed " + str(iterations) + " iterations without convergence.")
else:
    print("Completed " + str(iterations) + " iterations with convergence.")

# Find optimal policy
optimal_policy = [[None for j in range(25)] for i in range(50)]
for i in range(50):
    for j in range(25):
        if (i, j) not in grid_world.walls:
            pos_values = []
            value_action_dict = dict()
            x_up, y_up = i, j + 1
            x_down, y_down = i, j - 1
            x_left, y_left = i - 1, j
            x_right, y_right = i + 1, j

            all_actions = ['Up', 'Down', 'Left', 'Right']

            for action in all_actions:
                tmp = 0.0
                if action == 'Up':
                    if (x_up, y_up) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.8 * (-1 + discount * V[i, j])
                    else:
                        if (x_up, y_up) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[x_up, y_up])
                        else:
                            tmp += 0.8 * (0 + discount * V[x_up, y_up])

                    if (x_down, y_down) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_down, y_down) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_down, y_down])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_down, y_down])

                    if (x_left, y_left) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_left, y_left) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_left, y_left])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_left, y_left])

                    if (x_right, y_right) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_right, y_right) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_right, y_right])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_right, y_right])
                
                elif action == 'Down':
                    if (x_up, y_up) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_up, y_up) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_up, y_up])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_up, y_up])

                    if (x_down, y_down) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.8 * (-1 + discount * V[i, j])
                    else:
                        if (x_down, y_down) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[x_down, y_down])
                        else:
                            tmp += 0.8 * (0 + discount * V[x_down, y_down])

                    if (x_left, y_left) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_left, y_left) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_left, y_left])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_left, y_left])

                    if (x_right, y_right) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_right, y_right) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_right, y_right])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_right, y_right])

                elif action == 'Left':
                    if (x_up, y_up) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_up, y_up) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_up, y_up])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_up, y_up])

                    if (x_down, y_down) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_down, y_down) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_down, y_down])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_down, y_down])

                    if (x_left, y_left) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.8 * (-1 + discount * V[i, j])
                    else:
                        if (x_left, y_left) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[x_left, y_left])
                        else:
                            tmp += 0.8 * (0 + discount * V[x_left, y_left])

                    if (x_right, y_right) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_right, y_right) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_right, y_right])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_right, y_right])

                elif action == 'Right':
                    if (x_up, y_up) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_up, y_up) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_up, y_up])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_up, y_up])

                    if (x_down, y_down) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_down, y_down) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_down, y_down])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_down, y_down])

                    if (x_left, y_left) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.2/3 * (-1 + discount * V[i, j])
                    else:
                        if (x_left, y_left) == grid_world.goal:
                            tmp += 0.2/3 * (100 + discount * V[x_left, y_left])
                        else:
                            tmp += 0.2/3 * (0 + discount * V[x_left, y_left])

                    if (x_right, y_right) in grid_world.walls:
                        if (i, j) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[i, j])
                        else:
                            tmp += 0.8 * (-1 + discount * V[i, j])
                    else:
                        if (x_right, y_right) == grid_world.goal:
                            tmp += 0.8 * (100 + discount * V[x_right, y_right])
                        else:
                            tmp += 0.8 * (0 + discount * V[x_right, y_right])
                pos_values.append(tmp)
                value_action_dict[action] = tmp
            max_value = max(pos_values)
            pos_actions = []
            for action, value in value_action_dict.items():
                if value == max_value:
                    pos_actions.append(action)
            optimal_policy[i][j] = np.random.choice(pos_actions, 1)[0]


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
        if optimal_policy[i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, optimal_policy[i][j])
            all_arrows.append((x, y, dx, dy))

start_x, start_y = 1, 1
# Draw sample execution of optimal policy
test_agent = Agent(start_x, start_y, grid_world)

while test_agent.state_history[-1] != grid_world.goal:
    test_agent.execute_nominal_action(optimal_policy[test_agent.x][test_agent.y])

all_walls = list(grid_world.walls)
wall_color = ['r']*len(all_walls)

fig = plt.figure(figsize=(18, 7.5))
ax = fig.gca()
ax.set_xticks(np.arange(0, 51, 1))
ax.set_yticks(np.arange(0, 26, 1))
ax.set_xlim([-0.5, 49.5])
ax.set_ylim([-0.5, 24.5])
marker = markers.MarkerStyle(marker='s')
P = np.arange(25)
Q = np.arange(50)
all_points = np.dstack(np.meshgrid(P, Q)).reshape(-1, 2)
cmap = 'inferno'
X = list(range(1, len(test_agent.state_history) + 1))
scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, facecolor='w', edgecolors='k', marker=marker)
# for i in range(50):
#     for j in range(25):
#         plt.text(i, j, str(visit_count[i, j]))
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
scat3 = plt.scatter([test_agent.state_history[i][0] for i in range(len(test_agent.state_history))], [test_agent.state_history[i][1] for i in range(len(test_agent.state_history))], s=200, c=X, cmap=cmap, facecolor='w', edgecolors='k', marker=marker)

for i in range(len(all_arrows)):
    plt.arrow(all_arrows[i][0], all_arrows[i][1], all_arrows[i][2], all_arrows[i][3], length_includes_head=True, head_width=0.15, edgecolor='royalblue', facecolor='y')

norm = colors.Normalize(vmin=1, vmax=len(test_agent.state_history))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, label='Steps')
plt.savefig('c_test_run.png', bbox_inches='tight')
plt.show()





# Count visits    
visit_count = np.zeros((50, 25), dtype=int)

# Run episodes
episode_sz = 75
num_episodes = 200

mobile_agent = Agent(start_x, start_y, grid_world)

for i in range(num_episodes):
    for j in range(episode_sz):
        mobile_agent.execute_nominal_action(optimal_policy[mobile_agent.x][mobile_agent.y])

    for k in range(len(mobile_agent.state_history)):
        visit_count[mobile_agent.state_history[k][0], mobile_agent.state_history[k][1]] += 1

    mobile_agent.clear_history(start_x, start_y)

all_walls = list(grid_world.walls)
wall_color = ['r']*len(all_walls)

fig = plt.figure(figsize=(18, 7.5))
ax = fig.gca()
ax.set_xticks(np.arange(0, 51, 1))
ax.set_yticks(np.arange(0, 26, 1))
ax.set_xlim([-0.5, 49.5])
ax.set_ylim([-0.5, 24.5])
marker = markers.MarkerStyle(marker='s')
P = np.arange(25)
Q = np.arange(50)
all_points = np.dstack(np.meshgrid(P, Q)).reshape(-1, 2)
X = visit_count.reshape(-1, 1)
# X = np.log(X)
cmap = 'viridis'
scat3 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, facecolor='w', edgecolors='k', marker=marker)
scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], c=X[:, 0], cmap=cmap, s=200, facecolor='w', edgecolors='k', marker=marker)
# for i in range(50):
#     for j in range(25):
#         plt.text(i, j, str(visit_count[i, j]))
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)

norm = colors.Normalize(vmin=np.min(X), vmax=np.max(X))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, label='Visit count')
plt.savefig('c_' + str(episode_sz) + '.png', bbox_inches='tight')
plt.show()