from gym_minigrid.minigrid import *
from gym_minigrid.register import register

LOW = 0.2
HIGH = 1
CRASH_REWARD = -0.4
FAIL_HOME = -0.3
STEP_PENALTY = -0.005
#Helper class for keeping track of reward positions
class MazeEnd(Ball):
    reward = LOW
    def __init__(self, reward='low'):
        color = 'yellow'
        if reward == 'high':
            color = 'green'
            self.reward = HIGH
        super().__init__(color)

    def can_overlap(self):
        return True

    def set_high(self):
        self.reward = HIGH

class TMazeEnv(MiniGridEnv):

    #Actions that the agent is permitted to make
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, high_reward_end=0, size = 7, isHoming = False):
        self.agent_start_pos = (size // 2, 1)
        self.agent_start_dir = 1
        #Which maze end has the high reward. The rest have the low reward
        self.high_reward_end = high_reward_end
        self.reward_range = (-1, 1)
        self.actions = TMazeEnv.Actions
        self.isHoming = isHoming
        max_steps = 8
        if self.isHoming:
            max_steps = 15
        super().__init__(
            grid_size= size,
            max_steps = max_steps,
            agent_view_size = 3
        )


    def _gen_grid(self, width, height):
        #Setting the coordinates for each maze end
        self.LEFT_END = (1,self.height - 2)
        self.RIGHT_END = (self.height - 2, self.height - 2)
        self.MAZE_ENDS = [self.LEFT_END, self.RIGHT_END]
        self.middle = self.width // 2
        #Setting the coordinates for the turning points
        self.TURNING_POINTS = [(self.middle, self.width - 2)]
        self.reward = 0
        self.grid = Grid(width, height)

        #Here we are creating the landscape which looks like this
        #       X
        #       X
        #       X
        # X X X X X X X
        #
        for i in range(width):
            for j in range(height):
                self.put_obj(Lava(),i,j)

        for i in range (width):
            self.grid.set(i,height-2,None)
            self.grid.set(self.middle,i,None)
        self.mission = "Find the highest reward"

        x=0
        y=0
        w=width
        h=height
        self.grid.horz_wall(x, y, w,Lava)
        self.grid.horz_wall(x, y + h - 1, w,Lava)
        self.grid.vert_wall(x, y, h,Lava)
        self.grid.vert_wall(x + w - 1, y, h,Lava)

        #Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        #Set the maze ends with the correct rewards
        self.set_reward_pos(self.high_reward_end)
        #Set the home square
        self.put_obj(Goal(),self.agent_start_pos[0], self.agent_start_pos[1])

    def set_reward_pos(self, high_reward_end):
        self.high_reward_end = high_reward_end
        for i in range(len(self.MAZE_ENDS)):
            if i == high_reward_end:
                self.put_obj(MazeEnd('high'),self.MAZE_ENDS[i][0], self.MAZE_ENDS[i][1])
            else:
                self.put_obj(MazeEnd('low'), self.MAZE_ENDS[i][0], self.MAZE_ENDS[i][1])


    def step(self, action):
        #Some of the following code is copied from MiniGrid's step function
        self.step_count += 1
        reward = 0
        done = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            action = self.actions.forward

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            action = self.actions.forward

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            #If the agent crashes the episode ends and a negative reward is given
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
                reward = CRASH_REWARD
            # When we are simulating the homing T-maze task, the reward is delayed
            if fwd_cell != None and fwd_cell.type == 'ball':
                self.reward = fwd_cell.reward
                self.grid.set(self.agent_pos[0],self.agent_pos[1],None)
                if not self.isHoming:
                    done = True
                    reward = self.reward
                self.agent_dir = (self.agent_dir + 2) % 4

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        #Checking for number of step taken to punish the agent when it fails to return home
        if self.step_count >= self.max_steps:
            done = True
            reward = FAIL_HOME

        atHome = list(self.agent_pos) == list(self.agent_start_pos)
        atTurning = False
        for turningPoint in self.TURNING_POINTS:
            if list(turningPoint) == list(self.agent_pos):
                atTurning = True
        atMazeEnd = False
        for maze_end in self.MAZE_ENDS:
            if list(maze_end) == list(self.agent_pos):
                atMazeEnd = True
        #observation format = [is agent at home, is agent at turning point, is agent at maze end]
        obs = (float(atHome),float(atTurning),float(atMazeEnd))
        #Comment out for debugging purposes
        #print(obs)
        return obs, reward, done, {}

    def reset(self, reward_pos = -1):
        super().reset()
        atHome = list(self.agent_pos) == list(self.agent_start_pos)
        atTurning = False
        for turningPoint in self.TURNING_POINTS:
            if list(turningPoint) == list(self.agent_pos):
                atTurning = True
        atMazeEnd = False
        for maze_end in self.MAZE_ENDS:
            if list(maze_end) == list(self.agent_pos):
                atMazeEnd = True
        # observation format = [is agent at home, is agent at turning point, is agent at maze end]
        obs = (float(atHome), float(atTurning), float(atMazeEnd))
        #print(obs)
        if reward_pos in range(len(self.MAZE_ENDS)):
            self.set_reward_pos(reward_pos)
        return obs

    def _reward(self):
        return self.reward


class DoubleTMazeEnv(TMazeEnv):

    def __init__(self, high_reward_end = 0, isHoming = False):
        super().__init__(high_reward_end=high_reward_end,size=9, isHoming= isHoming)
        if self.isHoming:
            self.max_steps = 23
        else:
             self.max_steps = 12


    def _gen_grid(self, width, height):
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Find the highest reward"
        self.END1 = (1, self.height - 2)
        self.END2 = (1, 1)
        self.END3 = (self.width - 2, 1)
        self.END4 = (self.width - 2, self.height - 2)
        self.MAZE_ENDS = [self.END1, self.END2, self.END3, self.END4]
        self.middle = self.width // 2
        self.TURN1 = (self.middle, self.middle)
        self.TURN2 = (1, self.middle)
        self.TURN3 = (self.width-2, self.middle)
        self.TURNING_POINTS = [self.TURN1, self.TURN2, self.TURN3]
        self.reward = 0
        self.grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                self.put_obj(Lava(), i, j)

        x, y = self.agent_start_pos[0], self.agent_start_pos[1]
        self.put_obj(Goal(), x, y)
        while (y < self.TURN1[1]):
            y += 1
            self.grid.set(x, y, None)

        i = 0
        while x + i < self.TURN3[0]:
            i += 1
            self.grid.set(x+i, y, None)
            self.grid.set(x-i,y,None)

        j = 0
        while y+j < self.END4[1]:
            j+=1
            self.grid.set(x + i, y+j, None)
            self.grid.set(x - i, y+j, None)
            self.grid.set(x + i, y-j, None)
            self.grid.set(x - i, y-j, None)

        self.set_reward_pos(self.high_reward_end)
        self.put_obj(Goal(),self.agent_start_pos[0], self.agent_start_pos[1])

if __name__ == "__main__":
    #Matplotlib crashes on render for now but it is not going to be a problem
    tmaze = DoubleTMazeEnv()
    tmaze.set_reward_pos(1)
    tmaze.render()
    x = input("Press enter when finished")

#The following code is executed when this file is imported and is necessary for this environment to work
#with manual_control.py
class TMazeHoming(TMazeEnv):

    def __init__(self):
        super().__init__(isHoming = True)

