"""
An agent that learns to balance the gym's cartpole by:

    - encoding its state in a not too sparse small SDR 
    - using a pair of ValueMap-s to map danger scores for left-right actions. 

The algorithm is a form of simple Q-Learning: 

    - A pair of SDR value maps are used to read the fear of moving LEFT or RIGHT
    - At every step agent queries the two fear maps  and picks the least 
      scary move or moves randomly if both choices are similarily dangerous
    - Hooray (and lived happily ever after) happens when episode reaches maximum
      number of steps. 
    - Death happens when episode ends prematurely 
    - Only after death fear maps are updated with nightmares
    - It ignores gym's reward and penalizes the past N steps leading to death
      with an incremental penalty - such the closer to death the higher the penalty 
    - For example:
          if step[-N+5] was LEFT, then it adds 5 with SDR encoding state[-N+5] 
          to the LEFT fear map
    - So fear maps are incremented only by near death experiences, 
      the closer to death is an experience, a higher a fear is added.
   

Results:
    With the right parameters(*) it learns to survive in under 50 deaths.

    (*) sdr encoding should not be too large nor too sparse. 
    Each state parameter uses 7 bits out of 50 (14% sparsity)


Comments:

    The algorithm is similar to Q-Tables with the peculiarity that 
    a SDR ValueMap highlights significant correlations between 
    different properties within dataset. 
    e.g - the association of being far right AND having high 
    speed toward right will get a high fear score regardless of what 

    About complaining the algorithm uses a hand crafted reward instead of
    the one received from the cartpole's environment, 

    Sure, but it also means: 

    - reward policy is really important on how well&fast an agent learns
    - after all, no real-life RL agent will be deployed by simply training it with a sloppy policy
    - local, graded  rewards might be more useful than either very large sparse ones or tiny constant ones. 
    - a trully intelligent agent would learn how to pick its own local rewards 

    - The policies invented by nature - like fear - might be useful.

Copyright & Disclaimer: 
    Use the code as you wish, 
    Do not blame me for hurting your pole

"""
import numpy as np
from time import time
import gym, random, numba

from sdr_value_map import ValueCorrMap # The magic ingredient

SDR_SIZE  = 100  
SDR_BITS  =   4  # Number of ON bits for each state parameter
STATE_SIZE =  4  # How many parameters are within a state
LEFT,RIGHT = 0,1
NUM_EPISODES = 1000 
EMPTY_SDR = np.array([], dtype = np.uint32)

@numba.jit
def sdr_encoder(state, minims, maxims):
    sdr_starts  = ( np.arange(STATE_SIZE) * SDR_SIZE // STATE_SIZE).astype(np.uint32)
    sdr_widths  = np.array([sdr_starts[1]-SDR_BITS] * 4, dtype = np.uint32)
    sdr_bits = np.array([SDR_BITS]*STATE_SIZE, dtype = np.uint32)
    sdr = []
    ranges = (maxims - minims)
    values = state - minims
    # print(ranges, values)
    for i, val in enumerate(state):
        rang, val = ranges[i], values[i]
        width = int(sdr_widths[i] * val / rang)
        
        for k in range(sdr_bits[i]):
           sdr.append(width + k + sdr_starts[i])
    return np.array(sdr, dtype=np.uint32)

@numba.jit
def min_max_adjust(state, minims, maxims):
    where = np.where(state < minims)
    minims[where] = state[where]
    where = np.where(state > maxims)
    maxims[where] = state[where]

class SDR_Proxy_Player:
    # This is a wrapper over an actual player. 
    # its purpose is to convert the float representation to SDR
    # (SDR encoder). 
    # Its peculiarity vs usual scalar encoders is it dynamically
    # adjusts its range to map efficiently the state space into SDR space
    # 
    def __init__(self, player):
        # initial values for all state parameters are within -0.1 +0.1 range
        self.maxims =  0.1 * np.ones(4)
        self.minims = - self.maxims.copy()
        self.player = player

    def policy(self, state, reward, done):
        min_max_adjust(state, self.minims, self.maxims)
        sdr = sdr_encoder(state, self.minims, self.maxims)
        return self.player.policy(sdr,reward,done)

class AvoidantPlayer(): 
    """
    As I probably mentioned its survival strategy is fear:
    Every time it dies the state-action pairs preceding its death
      are saved into two fear maps: one ValueCorrMap encodes fear to move left
      and the other one to move right. 
      The closer to dying state the higher the fear level. 


    """

    def __init__(self):
        self.dangers = [ValueCorrMap(sdr_size = SDR_SIZE) for _ in (LEFT,RIGHT)]
        self.new_game()

    def new_game(self):
        self.steps = []
        self.reward = 0
        
    def least_danger(self,sdr): 
        # returns which action is least dangerous except t
        # when they-re so close that it picks randomly

        scores = [danger.score(sdr) for danger in self.dangers]
        e = 0.00001
        if  max(scores) / (min(scores)+e) < 1.01: 
            return random.randint(0,1)
        return scores[LEFT] > scores[RIGHT]

    def policy(self,sdr,reward,done):
        action = self.least_danger(sdr)
        self.steps.append((sdr, action))
        self.reward += reward
        if done :
            self.update_dangers()
            self.new_game()
        return action

    def update_dangers(self):
        # Now the game is ended 
        # This is where learning happens:
        if self.reward > 490:   # consider > 490 points as success
            print("yup! ", end = '')
            return              # nothing scary happened, don't record anything

        danger = 18 
        while danger and len(self.steps):
            sdr, action    = self.steps.pop()
            self.dangers[action].add(sdr, danger)
            danger -= 1
    

def cartpole_play(policy, n_episodes):
    cp = gym.make('CartPole-v1')
    for  ep in range(n_episodes):
        state = cp.reset()
        istate = state
        done = False
        reward = 0
        steps = []
        rtotal = 0
        while True: 
            action = policy(state, reward, done)
            if done: break
            state, reward, done, info = cp.step(action)
            cp.render()
            steps.append((action, state,  reward, done))
            rtotal += reward
        yield ep, rtotal,istate, steps       


player = SDR_Proxy_Player(AvoidantPlayer())
t = time()
total_steps = 0
hoorays = 0
for ep,total_reward,istate,steps in cartpole_play(player.policy, NUM_EPISODES):
    print(f"{ep+1:4d}: steps: {len(steps):3d}")
    if len(steps) == 500: 
        hoorays += 1
    total_steps += len(steps)
t = int((time() - t)*1000)
print(f"Play {total_steps} steps in {t}ms, hoorays:{hoorays}")
