#============== DEEP RL IN A BASIC MACROECONOMIC MODEL =================

#==== 0. DEPENDENCIES 
from numpy.core.fromnumeric import mean, var
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.sac.policies import MlpPolicy
import tensorflow as tf
from torch import nn
from stable_baselines3 import SAC, PPO, DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, base_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gym
import os
import yaml
pd.set_option('display.max_rows', 500)
sns.set_theme(palette="viridis")

from env import BaseEnv01, BaseEnv02


#==== 1. ENDOGENOUS INCOME 
# TRAINING
# Environment parameters
TOTAL_STEPS = 10
interest_rates = [1/99, 101/99, -49/99] # ß(1+r) = 1, 2, 0.5
models = [PPO]
income_paths = [1, 0, -1]
variables = ['Consumption', 'Labor', 'Savings']
trained_models = []

# Training parameters
TOTAL_TIMESTEPS = 1e5

log_path = os.path.join('training', 'logs')
save_path = os.path.join('training', 'saved_models')

def instantiate_env(interest_rate, total_steps):
    # Instantiating our environment
    env = BaseEnv02(interest_rate,
                    total_steps
                    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env.reset()
    return env

def instantiate_model(env):
    # Instantiating our agent
    model = PPO(env=env,
            gamma=0.99,     # Discount factor
            verbose=0,
            #tensorboard_log=log_path,
            policy='MlpPolicy',
            #**config['PPO_base_hyper']          
            )
    return model

def train_model(model,env,
                total_timesteps=TOTAL_TIMESTEPS,
                save_path=''
                ):
    # Callback
    eval_callback = EvalCallback(env,
                                eval_freq=1e5,
                                best_model_save_path=save_path,
                                verbose=0
                                )
    # Training
    model.learn(total_timesteps=total_timesteps,
                callback=eval_callback
                )
    return model

# Looping for Instantiating and training
for i in range(len(interest_rates)):
    # Path
    best_model_path = os.path.join(save_path, f'PPO_base_env02_r_{interest_rates[i]:.2f}')
    os.makedirs(best_model_path, exist_ok=True)
    
    env = instantiate_env(interest_rates[i], TOTAL_STEPS)
    model = instantiate_model(env)
    trained_model = train_model(model,env,
                                total_timesteps=TOTAL_TIMESTEPS,
                                save_path=best_model_path
                                )
    trained_models.append(trained_model)


# TESTING 
def test_model(model_path, env):
    # Loading our agent
    model = PPO.load(model_path,
                    env=env
                    )
    #Testing
    mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=10, deterministic=True)
    print(f'\nMean reward: {mean_reward}')
    print('=========\n')
    obs = env.reset()
    for i in range(TOTAL_STEPS):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print(action)
    return

# Looping for Testing
for i in range(len(interest_rates)):
    # Path
    best_model_path = os.path.join(save_path, f'PPO_base_env02_r_{interest_rates[i]:.2f}', 'best_model')

    env = instantiate_env(interest_rates[i], TOTAL_STEPS)
    test_model(best_model_path, env)

# GENERATING PATHS/PREPROCESSING FOR PLOTTING
def fill_list_with_blanks(x, episode_length):
    for _ in range(episode_length):
        if len(x)<episode_length:
            x.append(None)
    return x

def get_paths(model_path, env):
    paths = [ [] for _ in range(3) ] #[Consumption, Labor, Savings]
    # Loading our agent
    model = PPO.load(model_path,
                    env=env
                    )
    obs = env.reset()
    done = False
    paths[0].append(None)     #Consumption
    paths[1].append(None)     #Labor
    paths[2].append(0)        #Savings

    for t in range(TOTAL_STEPS):  #To account for termination step
        action, _states = model.predict(obs, deterministic=False)       # Because of bootstrap
        obs, reward, done, info = env.step(action)
        paths[0].append(action[0][0])                # Double wrapped in DummyVecEnv
        paths[1].append(action[0][1])
        if t==TOTAL_STEPS:
            paths[2].append(info[0]['terminal_observation'][0])
        else:
            paths[2].append(obs[0][0])  

    paths = [fill_list_with_blanks(path, TOTAL_STEPS+1) for path in paths]
    return paths

# SIMULATE 100 INDIVIDUALS 
df_simulation = pd.DataFrame()
NUM_SIMULATIONS = 100
for s in range(NUM_SIMULATIONS):
    # Get consumption/labor/savings paths for each individual
    df = [None] * len(interest_rates)
    for i in range(len(interest_rates)):
        # Path
        best_model_path = os.path.join(save_path, f'PPO_base_env02_r_{interest_rates[i]:.2f}', 'best_model')

        env = instantiate_env(interest_rates[i], TOTAL_STEPS)
        paths = get_paths(best_model_path, env)
        # Appending to dataframe at the individual level 
        df[i] = pd.DataFrame(np.array(paths).T)
        df[i].columns = [f'Consumption_{interest_rates[i]:.2f}', f'Labor_{interest_rates[i]:.2f}', f'Savings_{interest_rates[i]:.2f}']

    # Concatenating and reshaping
    df_concat = pd.concat(df, axis=1)
    df_concat['time_step'] = df_concat.index
    df_long = pd.wide_to_long(df_concat, variables, i = 'time_step', j='int_rate', sep='_', suffix='(.*)').reset_index()
    df_long = pd.melt(df_long, id_vars=['time_step', 'int_rate'], value_vars=variables,
                            var_name='var_type', value_name='Value' )    #[time_step, int_rate, var_type, Value]
    df_long['ID'] = s   # Individual identifier
    df_simulation = df_simulation.append(df_long)

df_simulation = df_simulation[['ID', 'time_step', 'int_rate', 'var_type', 'Value']]

# PLOTTING
palette = sns.color_palette("mako_r")

g = sns.FacetGrid(df_simulation, col='int_rate', hue='var_type')
g.map(sns.lineplot,
        'time_step', 'Value'
        #palette = palette
        
    )
plt.gca().invert_yaxis()
plt.show()

# Falta compararlo con los paths de Euler, y compararar las utilidades descontadas
## ACA ME QUEDE










# Final dataframe
column_names = ['ID', 'step', 'var_type', 'int_rate', 'Model', 'Value']

df = pd.DataFrame(None, index=np.arange(NUM_OBS*(TOTAL_STEPS+1)*len(variables)*len(interest_rates)*len(models)), columns=column_names)

ID = pd.DataFrame([x for x in range(NUM_OBS)], columns=['ID'])
step = pd.DataFrame([x for x in range(TOTAL_STEPS+1)], columns=['step'])
variables = pd.DataFrame(['Consumption', 'Labor', 'Savings'], columns=['variables'])
interest_rates = pd.DataFrame([1/99, 101/99, -49/99], columns=['interest_rates']) # ß(1+r) = 1, 2, 0.5
models = pd.DataFrame([PPO], columns=['models'])

data = []
for i in range(ID):
    data.append()

matrix = np.zeros_like(df)

y = [a for a in range(5)]          
asd = [[x for x in range(2)] for _ in y]
np.array(asd).flatten()
df.unstack().reset_index() ## USAR ESTO !!! osea crear la base normal y luego hacer este reshape long para los plots

for int_r in range(len(interest_rates)):
    # Instantiating our environment
    env = BaseEnv02(interest_rate=interest_rates[int_r]
                    ,total_steps=TOTAL_STEPS
                    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env.reset()
    # Loading our agent
    best_model_path = os.path.join(save_path, f'PPO_base_env02_r_{interest_rates[int_r]}', 'best_model')
    model = PPO.load(best_model_path
        , env=env)

    for obs in range(NUM_OBS):
        start = (obs)*(TOTAL_STEPS+1)
        end = (obs+1)*(TOTAL_STEPS+1)
        df.iloc[start:end, 0] = obs    # ID

        consumption = [0]           # For now
        labor = [0]                 # For now
        savings = [0]
        obs = env.reset()
        done = False

        for t in range(TOTAL_STEPS+1):          ### +1?
            df.iloc[start+t, 1] = t   # Time step
            action, _states = model.predict(obs, deterministic=False)       # Because of bootstrap
            obs, reward, done, info = env.step(action)
            consumption.append(action[0][0])                # Double wrapped in DummyVecEnv
            labor.append(action[0][1])
            if t==TOTAL_STEPS:
                savings.append(info[0]['terminal_observation'][0])
            else:
                savings.append(obs[0][0])  

        consumption = fill_list_with_blanks(consumption, TOTAL_STEPS+1)
        labor = fill_list_with_blanks(labor, TOTAL_STEPS+1)
        savings = fill_list_with_blanks(savings, TOTAL_STEPS+1)
        
        # Adding to df
        df.iloc[start:end, pos] = np.asarray(consumption)
        pos += 1
        df.iloc[start:end, pos] = np.asarray(labor)
        pos += 1
        df.iloc[start:end, pos] = np.asarray(savings)
        pos += 1


for s in range(NUM_OBS):
    start = (s)*(TOTAL_STEPS+1)
    end = (s+1)*(TOTAL_STEPS+1)
    df.iloc[start:end, 0] = s    # ID
    pos = 2
    for i in range(len(interest_rates)):
        # Instantiating our environment
        env = BaseEnv02(interest_rate=interest_rates[i]
                        ,total_steps=TOTAL_STEPS
                        )
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env.reset()

        # Loading our agent
        best_model_path = os.path.join(save_path, f'PPO_base_env02_r_{interest_rates[i]}', 'best_model')
        model = PPO.load(best_model_path
            , env=env)
        
        consumption = [0]           # For now
        labor = [0]                 # For now
        savings = [0]
        obs = env.reset()
        done = False
        cont = 0
        df.iloc[start+cont, 1] = cont
        while not done:
            cont +=1
            df.iloc[start+cont, 1] = cont
            action, _states = model.predict(obs, deterministic=False)       # Because of bootstrap
            obs, reward, done, info = env.step(action)
            consumption.append(action[0][0])                # Double wrapped in DummyVecEnv
            labor.append(action[0][1])
            if cont==TOTAL_STEPS:
                savings.append(info[0]['terminal_observation'][0])
            else:
                savings.append(obs[0][0])

        consumption = fill_list_with_blanks(consumption, TOTAL_STEPS+1)
        labor = fill_list_with_blanks(labor, TOTAL_STEPS+1)
        savings = fill_list_with_blanks(savings, TOTAL_STEPS+1)
        
        # Adding to df
        df.iloc[start:end, pos] = np.asarray(consumption)
        pos += 1
        df.iloc[start:end, pos] = np.asarray(labor)
        pos += 1
        df.iloc[start:end, pos] = np.asarray(savings)
        pos += 1
df.head(20)
df.to_csv('df_base_env02_c.csv')

df.pivot_table(index=["ID", "step"], 
                    columns='class', 
                    values='grade')

# PLOTTING

palette = sns.color_palette("mako_r", 6)
y_plot = list([df.columns[4], df.columns[5]])
list('PPO_')
sns.lineplot(
    data=df, x="step", y=y_plot,
    palette=palette
)
plt.show()













#%% ################## EXOGENOUS WAGE ###########################

TOTAL_STEPS = 5
interest_rates = [1/99] # ß(1+r) = 1, 
models = [PPO]
income_paths = [1, 0, -1]

log_path = os.path.join('training', 'logs')
save_path = os.path.join('training', 'saved_models')

for i in range(len(interest_rates)):
    for j in range(len(models)):
        for k in range(len(income_paths)):

            # Instantiating our environment
            env = BaseEnv(interest_rate=interest_rates[i]
                         ,income_path=income_paths[k]
                         ,total_steps=TOTAL_STEPS
                            )
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            env.reset()

            # Instantiating our agent
            model = models[j](env=env
                    ,gamma=0.99     # Discount factor
                    ,verbose=0
                    ,tensorboard_log=log_path
                    ,policy='MlpPolicy'
                    #**config['PPO_base_hyper']          
                    )
            
            # Training our model
                # Path
            best_model_path = os.path.join(save_path, f'Model_{models[j]}_base_env_r_{interest_rates[i]}_{income_paths[k]}')
            os.makedirs(best_model_path, exist_ok=True)

            eval_callback = EvalCallback(env
                                        ,eval_freq=10000
                                        ,best_model_save_path=best_model_path
                                        ,verbose=0)
                # Training
            if ((models[j]==PPO) or (models[j]==A2C)): model.learn(total_timesteps=5e6
                                                                    ,callback=eval_callback    )
            # if (models[j]==SAC) or (models[j]==DDPG): model.learn(total_timesteps=2e5)
            #model.save(os.path.join(save_path, f'Model_{models[j]}_base_env_r_{interest_rates[i]}_{income_paths[k]}'))


################## TESTING ###########################

interest_rates = [1/99] # ß(1+r) = 1, 
models = [PPO]
income_paths = [1, 0, -1]

log_path = os.path.join('training', 'logs')
save_path = os.path.join('training', 'saved_models')

for i in range(len(interest_rates)):
    for j in range(len(models)):
        for k in range(len(income_paths)):
            # Instantiating our environment
            env = BaseEnv(interest_rate=interest_rates[i]
                         ,income_path=income_paths[k]
                         ,total_steps=TOTAL_STEPS
                            )
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            env.reset()
            
            # Loading our agent
            best_model_path = os.path.join(save_path, f'Model_{models[j]}_base_env_r_{interest_rates[i]}_{income_paths[k]}','best_model')
            model = models[j].load(best_model_path
                , env=env)
            mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=10, deterministic=True)

# Enjoy trained agent
obs = env.reset()
for i in range(TOTAL_STEPS):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)


################# PREPROCESSING FOR PLOTTING ##################

interest_rates = [1/99] # ß(1+r) = 1, 
models = [PPO]
income_paths = [1, 0, -1]

log_path = os.path.join('training', 'logs')
save_path = os.path.join('training', 'saved_models')
episode_length=TOTAL_STEPS+1

def fill_list_with_blanks(x, episode_length):
    for _ in range(episode_length):
        if len(x)<episode_length:
            x.append(None)
    return x

column_names = ['PPO_1_consumption', 'PPO_1_savings',
                'PPO_0_consumption', 'PPO_0_savings',
                'PPO_-1_consumption', 'PPO_-1_savings']
df = pd.DataFrame(columns=column_names)


pos = 0
for i in range(len(interest_rates)):
    for j in range(len(models)):
        for k in range(len(income_paths)):
            # Instantiating our environment
            env = BaseEnv(interest_rate=interest_rates[i]
                         ,income_path=income_paths[k]
                         ,total_steps=TOTAL_STEPS
                            )
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            env.reset()

            # Loading our agent
           # model = models[j].load(os.path.join(save_path, f'Model_{models[j]}_base_env_r_{interest_rates[i]}_{income_paths[k]}')
             #   , env=env)
            best_model_path = os.path.join(save_path, f'Model_{models[j]}_base_env_r_{interest_rates[i]}_{income_paths[k]}','best_model')
            model = models[j].load(best_model_path
                , env = env)
            consumption = [None]
            savings = [0]
            obs = env.reset()
            done = False
            cont = 0
            while not done:
                cont +=1
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                consumption.append(action[0][0])
                if cont==TOTAL_STEPS:
                    savings.append(info[0]['terminal_observation'][0])
                else:
                    savings.append(obs[0][0])

            consumption = fill_list_with_blanks(consumption,episode_length)
            savings = fill_list_with_blanks(savings,episode_length)
            
            # Adding to df
            df.iloc[: , pos] = np.asarray(consumption)
            pos += 1
            df.iloc[: , pos] = np.asarray(savings)
            pos += 1

df.to_csv('df_base_env.csv')

