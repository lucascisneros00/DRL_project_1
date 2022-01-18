from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
import math

class BaseEnv01(Env):
    """
    Description:
        An agent receives a monotonic income path (y) at each step of his life (episode
        =T). He can save it (b) (and get a return (1+r)b the next step) or consume 
        it (c). Note that at each step, c_t + b_t = y_t + (1+r)b_t-1. The agent 
        must pay his debt in the last period, thus b_100=0. At each step, the 
        agent has a Reward (utility function) U(c_t), which he intends to maximize.

    Observation (state variables):
        Type: Box(1)
        Num     Observation                Min                     Max
        0       Savings (b_t)             -1 (debt)               +1 (savings)

    Actions (control variables):
        Type: Box(1)
        Num   Action                        Min                    Max
        0     Consumption (c_t)             0                      +1 

    Reward:
        Reward is U(c_t) for every step taken, including the termination step.
        U(c_t) = ln(1+c_t)

    Starting State:
        The agents starts his life with no debt/savings (b_0=0). 

    Episode Termination:
        Episode length is greater than T.
    """

    def __init__(self, interest_rate=1/99, income_path=0, total_steps=2):
        self.min_action = 0
        self.max_action = 1
        self.min_savings = -1
        self.max_savings = 1
        self.total_steps = total_steps
        self.path = income_path        # =0: flat, =1: increasing, =-1: decreasing
        self.interest_rate = interest_rate
        
        # Continous action space
        self.action_space = Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        # Continous observation space
        self.observation_space = Box(
            low=self.min_savings,
            high=self.max_savings,
            shape=(1,),
            dtype=np.float32
        )  
        
        self.seed()
        self.state = None
        # Set amount of periods the agent will live
        self.steps_left = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        #####
        # Helper function
        def get_income(self):  
            if self.path == -1: a = 1
            if self.path == 0: a = 0.5
            if self.path == 1: a = 0
            b = self.path/(self.total_steps-1)
            x = self.steps
            return a + b*x
        ######

        savings = self.state
        income = get_income(self)

        # Apply action
        self.steps += 1
        savings = (1+self.interest_rate)*savings + income - action

        # Check if done
        done = bool(
            self.steps>=self.total_steps # Episode is over
            or savings>self.max_savings
            or savings<self.min_savings
        )

        # Calculate reward
        reward = math.log(1+action)
        if done:
            if self.steps<self.total_steps:
                reward = -10*self.total_steps  # The agent didn't finish his life period
            else:
                if savings<0: reward = -10*self.total_steps  # The agent MUST pay his debts at the end of his lifetime (No-Ponzi)

        self.state = (savings)
        info = {}
        # Return step information
        return np.array(self.state), reward, done, info

    def reset(self):
        # Reset savings
        self.state = self.np_random.uniform(low=-.0, high=.0, size=(1,))
        # Reset lifetime
        self.steps = 0
        return np.array(self.state)

    def render(self):
        pass




class BaseEnv02(Env):
    """
    Description:
        An agent lives for T periods, consumes c_t and works l_t. He receives a wage
        w for his work. He can choose to save part of his income b_t and get a return
        (1+r)b_t in the next period. Note that at each period, their budget constraint
        is c_t + b_t = w*l_t + (1+r)b_t-1.
        The agent has a Reward (utility function) U(c_t), which he intends to maximize.
        He receives utility from consuming and disutility from working.

    Observation (state variables):
        Type: Box(1)
        Num     Observation                Min                     Max
        0       Savings (b_t)             -1 (debt)               +1 (savings)

    Actions (control variables):
        Type: Box(1)
        Num   Action                        Min                    Max
        0     Consumption (c_t)             0                      +1 
        1     Labor (l_t)                   0                      +1

    Reward:
        Reward is U(c_t, l_t) at each period, including the termination step.
        U(c_t,l_t) = c_t^(1-eta) / (1-eta) - l_t^(1-psi)/(1-psi)

    Starting State:
        The agents starts and ends his life with no debt/savings (b_0=b_T=0). 

    Episode Termination:
        Episode length is greater than T.
    """

    def __init__(self, interest_rate=1/99, total_steps=10):
        self.min_consumption = 0
        self.max_consumption = 1
        self.min_labor = 0
        self.max_labor = 1

        self.min_savings = -total_steps
        self.max_savings = total_steps

        self.eta = 0.8
        self.psi = 0.4

        self.total_steps = total_steps
        self.wage = 1
        self.interest_rate = interest_rate
        
        # Continous action space
        self.action_space = Box(
            low=np.array([self.min_consumption, self.min_labor], dtype=np.float32),
            high=np.array([self.max_consumption, self.max_labor], dtype=np.float32),
            dtype=np.float32
        )

        # Continous observation space
        self.observation_space = Box(
            low=self.min_savings,
            high=self.max_savings,
            shape=(1,),
            dtype=np.float32
        )  
        
        self.seed()
        self.state = None
        # Set amount of periods the agent will live
        self.steps_left = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        consumption, labor = action
        savings = self.state

        # Apply action
        self.steps += 1
        savings = (1+self.interest_rate)*savings + self.wage*labor - consumption

        # Check if done
        done = bool(
            self.steps>=self.total_steps # Episode is over
            or savings>self.max_savings
            or savings<self.min_savings
        )

        # Calculate reward
        reward = consumption**(1-self.eta) / (1-self.eta) - labor**(1-self.psi)/(1-self.psi)
        if done:
            if self.steps<self.total_steps:
                reward = -50*self.total_steps  # The agent didn't finish his life period
            else:
                if savings<0: reward = -50*self.total_steps  # The agent MUST pay his debts at the end of his lifetime (No-Ponzi)

        self.state = (savings)
        info = {}
        # Return step information
        return np.array(self.state), reward, done, info

    def reset(self):
        # Reset savings
        self.state = self.np_random.uniform(low=-.0, high=.0, size=(1,))
        # Reset lifetime
        self.steps = 0
        return np.array(self.state)

    def render(self):
        pass
