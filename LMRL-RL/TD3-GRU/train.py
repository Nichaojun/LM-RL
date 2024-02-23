import time
import os
import numpy as np
import torch
from env import GazeboEnv
from buffer import ReplayBuffer
# from gru_net import TD3
from td3_net import TD3
# from attention_net import TD3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # 设备
seed = 0                                                                        # 随机种子
eval_freq = 5e3                                                                 # 间隔多少个step做一次评估
eval_ep = 10                                                                     # 做多少次评估 
max_ep_step = 500                                                               # 单个episode的最大步数
max_timesteps = 5e6                                                             # 总共的最大步数
save_models = True                                                              # 是否保存模型
max_his_len = 10                                             
expl_noise = 1.0                                                                # 动作噪音相关参数
expl_decay_steps = 500000
expl_min = 0.1
save_reward = -9999
discount = 0.99999
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2                                                                 # actor网络更新频率
file_name = "TD3_velodyne"
random_near_obstacle = False                                                     # 是否在障碍物附近采取随机动作

#* 2. 创建存储模型的文件夹
if not os.path.exists("./results"): os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"): os.makedirs("./pytorch_models")

#* 3. 环境初始化
env = GazeboEnv('multi_robot_scenario.launch')
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)


network = TD3()
replay_buffer = ReplayBuffer(seed)

#* 6. 初始化训练参数
state_dim = 24
action_dim = 2
max_action = 1

count_rand_actions = 0
random_action = []

evaluations = []
timestep = 0
timesteps_since_eval = 0
episode_num = 0
epoch = 1
done = True
total_reward = 0
collide = False
start_update_timestep = 100
network_action_timestep = 10000

def evaluate(network, eval_episodes = 10, epoch=0, max_his_len = 10):
    avg_reward = 0.
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()                                                                                         # state: numpy(1, 23)
        done = False
        EP_HS = np.zeros([max_his_len, 24])
        EP_HS[::] = list(state)
        EP_HL = 0

        while not done and count < 501:
            action = network.get_action(state, EP_HS, EP_HL)
            a_in = [(action[0] + 1) / 2, action[1]]
            
            next_state, reward, done, _ = env.step(a_in, count-1)
            if EP_HL == max_his_len:
                EP_HS[:(max_his_len-1)] = EP_HS[1:]
                EP_HS[max_his_len-1] = list(state)
            else:
                if EP_HL > 1:
                    EP_HS[-(EP_HL+1)] = list(state)
                EP_HL += 1
            
            state = next_state
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1

    avg_reward /= eval_episodes
    avg_col = col/eval_episodes
    print("..............................................")
    print("Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f" % (eval_episodes, epoch, avg_reward, avg_col))
    print("..............................................")
    return avg_reward

state = env.reset()
episode_reward = 0
episode_timesteps = 0
episode_num += 1
EP_HS = np.zeros([max_his_len, state_dim])
EP_HS[::] = list(state)
EP_HL = 0


while timestep < max_timesteps:
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
    # according to state choose action 5000
    if timestep >= network_action_timestep:
        action = network.get_action(state, EP_HS, EP_HL)
        action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)
    else: 
        action = np.random.uniform(-1,1,2)

    if random_near_obstacle:
        if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1:
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)
        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    a_in = [(action[0] + 1) / 2, action[1]]


    # action into env, get state
    next_state, reward, done, target = env.step(a_in, episode_timesteps)
    episode_reward += reward
    
    episode_timesteps += 1                                                                  # episode的timestep
    timestep += 1                                                                           # 总的timestep
    timesteps_since_eval += 1

    if episode_timesteps == max_ep_step:
        done = False

    replay_buffer.add(state, action, reward, done, next_state)

    if EP_HL == max_his_len:
        EP_HS[:(max_his_len-1)] = EP_HS[1:]
        EP_HS[max_his_len-1] = list(state)
    else:
        if EP_HL > 1:
            EP_HS[-(EP_HL+1)] = list(state)
        EP_HL += 1

    state = next_state
    

    if done or episode_timesteps == max_ep_step:
        
        if timestep >= network_action_timestep:
            print('\033[1;45m Actor Action Update \033[0m', 'episode_reward:', round(episode_reward, 2), 'evaluation:', timesteps_since_eval)
        else:
            if timestep >= start_update_timestep:
                print('\033[1;45m Random Action Update \033[0m', 'evaluation:', timesteps_since_eval)
            else:
                print('\033[1;46m Data Collection \033[0m')


    # if timestep >= start_update_timestep and timestep % 50 == 0:
        if timestep >= start_update_timestep:
            network.train(replay_buffer, 50, 
                        discount, tau, policy_noise, noise_clip, policy_freq)
        
        state = env.reset()                                                                                         # state: numpy(1, 23)
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        EP_HS = np.zeros([max_his_len, state_dim])
        EP_HS[::] = list(state)
        EP_HL = 0

    if timesteps_since_eval >= eval_freq:
        print("================= Validating =================")
        timesteps_since_eval %= eval_freq
        tmp_reward = evaluate(network, eval_ep, epoch)
        evaluations.append(tmp_reward)
        if tmp_reward >= save_reward:
            save_reward = tmp_reward
            network.save(file_name, directory="./best_models")
        network.save(file_name, directory="./final_models")
        np.save("./results/%s" % (file_name), evaluations)
        epoch += 1
