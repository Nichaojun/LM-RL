from env import GazeboEnv
import numpy as np
import time
file_name = "model"                                  # agent文件所在的位置

id = 2
if id == 1:
    from td3_net import TD3
    network = TD3()
    network.load(file_name, directory="./baseline_model/final_models")

if id == 2:
    from attention_net import TD3
    network = TD3()
    network.load(file_name, directory="./attention_model/final_models")

if id == 3:
    from gru_net import TD3
    network = TD3()
    network.load(file_name, directory="./gru_model/final_models")


env = GazeboEnv('multi_robot_scenario.launch')
time.sleep(20)
avg_reward = 0
col = 0
eval_episodes = 10
max_his_len = 10

avg_ten = 0
avgbox = []

for i in range(eval_episodes):
    count = 0
    episode_reward = 0
    state = env.reset()                                                                                  # state: numpy(1, 23)
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
        episode_reward += reward
        avg_ten += reward
        count += 1
        if reward < -90:
            col += 1

avg_reward /= eval_episodes
avg_col = col/eval_episodes
print("..............................................")
print("Overver %i test episdoe, Average Reward: %f, Average Collision: %f" % (eval_episodes, avg_reward, avg_col))
print("..............................................")
