import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    
    def __init__(self, state_dim = 24):
        super(Actor, self).__init__()
        self.cat_len = 128
        # 历史state
        self.HFC1 = nn.ModuleList()
        self.LSTM = nn.ModuleList()
        self.HFC2 = nn.ModuleList()
        self.HFC1 += [nn.Linear(state_dim, 256), nn.ReLU()]
        self.LSTM += [nn.GRU(256, 256, batch_first=True)]

        self.w_omega = nn.Parameter(torch.Tensor(256, 256))
        self.u_omega = nn.Parameter(torch.Tensor(256, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.HFC2 += [nn.Linear(256, self.cat_len), nn.ReLU(), nn.Dropout()]

        # 当前state
        self.CFC = nn.ModuleList()
        self.FinalFC = nn.ModuleList()
        self.CFC += [nn.Linear(state_dim, 3*self.cat_len), nn.ReLU(), nn.Dropout()]
        self.FinalFC += [nn.Linear(4*self.cat_len, 2), nn.Tanh()]

    def forward(self, state, his_state, his_len):
        
        # 历史数据
        his_input = his_state                                                       # [1, 10, 22]                                                
        for layer in self.HFC1: his_input = layer(his_input)                        # [1, 10, 256]
        for layer in self.LSTM: his_input, _ = layer(his_input)                     # [1, 10, 256]
        
        # 开始attention
        x = his_input
        u = torch.tanh(torch.matmul(x, self.w_omega))                               # [1, 10, 256]
        att = torch.matmul(u, self.u_omega)                                         # [1, 10, 1]
        att_score = F.softmax(att, dim=1)                                           # [1, 10, 1]
        scored_x = x * att_score                                                    # [1, 10, 256]
        feat = torch.sum(scored_x, dim=1)                                           # [1, 256]
        for layer in self.HFC2: feat = layer(feat)                                  # [1, 128]                            
        his_output = feat

        # 当前数据
        cur_input = state                                                            
        for layer in self.CFC: cur_input = layer(cur_input)                          # [1, 384]                              
        cur_output = cur_input

        # 合并历史和当前数据
        final_input = torch.cat([his_output, cur_output], dim = -1)                  # [1, 512]
        for layer in self.FinalFC: final_input = layer(final_input)                  # [1, 2]   
        action = final_input
        
        return action

class Critic(nn.Module):

    def __init__(self, state_dim = 24, action_dim = 2):
        super(Critic, self).__init__()
        self.cat_len = 128
        # 历史数据: state
        self.HS1 = nn.ModuleList()
        self.HS_LSTM = nn.ModuleList()
        self.HS2 = nn.ModuleList()
        self.HS1 += [nn.Linear(state_dim, 256), nn.ReLU()]
        self.HS_LSTM += [nn.GRU(256, 256, batch_first=True)]
        
        self.w_omega = nn.Parameter(torch.Tensor(256, 256))
        self.u_omega = nn.Parameter(torch.Tensor(256, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.HS2 += [nn.Linear(256, self.cat_len), nn.ReLU()]

        # 当前数据: state
        self.SA = nn.ModuleList()
        self.SA += [nn.Linear(state_dim + action_dim, 3*self.cat_len), nn.ReLU()]

        # 最后
        self.Final = nn.ModuleList()
        self.Final += [nn.Linear(4*self.cat_len, 64),nn.ReLU(), nn.Linear(64 , 1), nn.Identity()]

        # =================================================================
        # 历史数据: state
        self._HS1 = nn.ModuleList()
        self._HS_LSTM = nn.ModuleList()
        self._HS2 = nn.ModuleList()
        self._HS1 += [nn.Linear(state_dim, 256), nn.ReLU()]
        self._HS_LSTM += [nn.GRU(256, 256, batch_first=True)]
        
        self._w_omega = nn.Parameter(torch.Tensor(256, 256))
        self._u_omega = nn.Parameter(torch.Tensor(256, 1))
        nn.init.uniform_(self._w_omega, -0.1, 0.1)
        nn.init.uniform_(self._u_omega, -0.1, 0.1)

        self._HS2 += [nn.Linear(256, self.cat_len), nn.ReLU()]

        # 当前数据: state
        self._SA = nn.ModuleList()
        self._SA += [nn.Linear(state_dim + action_dim, 3*self.cat_len), nn.ReLU()]

        # 最后
        self._Final = nn.ModuleList()
        self._Final += [nn.Linear(4*self.cat_len, 64),nn.ReLU(), nn.Linear(64, 1), nn.Identity()]

    def forward(self, state, action, his_state, his_action, his_len):

        # 历史数据
        hs = his_state
        _hs = his_state
        for layer in self.HS1: hs = layer(hs)
        for layer in self._HS1: _hs = layer(_hs)
        for layer in self.HS_LSTM: hs, _ = layer(hs)
        for layer in self._HS_LSTM: _hs, _ = layer(_hs)

        # 开始 attention
        x = hs
        _x = _hs
        u = torch.tanh(torch.matmul(x, self.w_omega))
        _u = torch.tanh(torch.matmul(_x, self._w_omega))

        att = torch.matmul(u, self.u_omega)
        _att = torch.matmul(_u, self._u_omega)

        att_score = F.softmax(att, dim=1)
        _att_score = F.softmax(_att, dim=1)

        scored_x = x * att_score
        _scored_x = _x * _att_score

        feat = torch.sum(scored_x, dim=1) 
        _feat = torch.sum(_scored_x, dim=1)     

        for layer in self.HS2: feat = layer(feat)
        for layer in self._HS2: _feat = layer(_feat)
        hist_out = feat
        _hist_out = _feat
        
        # current state action
        sa = torch.cat([state, action], dim = -1)
        _sa = torch.cat([state, action], dim = -1)

        for layer in self.SA: sa = layer(sa)
        for layer in self._SA: _sa = layer(_sa)

        cur_out = sa
        _cur_out = _sa

        # final
        final = torch.cat([hist_out, cur_out], dim = -1)
        _final = torch.cat([_hist_out, _cur_out], dim = -1)

        for layer in self.Final: final = layer(final)
        for layer in self._Final: _final = layer(_final)

        q1 = torch.squeeze(final, -1)
        q2 = torch.squeeze(_final, -1)
        

        return q1, q2


class TD3(object):
    
    def __init__(self):
        
        # actor, actor_target初始化
        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # critic, critic_target初始化
        self.critic = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = 1

    # 这个函数只会在A过程(episode的每一步)中调用
    def get_action(self, state, his_state, his_len):
        his_state = torch.Tensor(his_state).view(1, his_state.shape[0], his_state.shape[1]).float().cuda()      # A: [1, 10 ,22]
        his_len = torch.Tensor([his_len]).float().cuda()                                                        # A: [0-10]
        # print('his_state', his_state)
        episode_per_step_action = self.actor(torch.as_tensor(state, dtype=torch.float32).view(1, -1).cuda(), 
                                    his_state, his_len).cpu().data.numpy().flatten()
        return episode_per_step_action                                                                          # A: [1, 2]                                 

    def train(self, replay_buffer, iterations, discount = 1, 
                tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
        
        #* 在一个episode中多少步，就更新多少次参数                                            
        for it in range(iterations):
            
            # 从 replay buffer 里面选数据
            batch = replay_buffer.sample_batch()
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action = batch['action'].to(device)
            reward = batch['reward'].to(device)
            done = batch['done'].to(device)
            h_state = batch['h_state'].to(device)
            h_next_state = batch['h_next_state'].to(device)
            h_action = batch['h_action'].to(device)
            h_next_action = batch['h_next_action'].to(device)
            h_state_len = batch['h_state_length'].to(device)
            h_next_state_len = batch['h_next_state_length'].to(device)            

            current_Q1, current_Q2 = self.critic(state, action, h_state, h_action, h_state_len)

            #* 从target_actor中获取下一个动作
            next_action = self.actor_target(next_state, h_next_state, h_next_state_len)
            noise = action.data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            #* 从target_critic中得到target_Q值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, h_next_state, 
                                                        h_next_action, h_next_state_len)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            #* 从critic中得到Q1, Q2
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            #! critic 参数更新
            self.critic_optimizer.zero_grad()
            # print('critic_loss:', critic_loss.item()[0])
            critic_loss.backward()
            self.critic_optimizer.step()

            #! actor 参数更新
            if it % policy_freq == 0:

                actor_loss, _ = self.critic(state, self.actor(state, h_state, h_state_len), 
                            h_state, h_action, h_state_len)

                actor_loss = -actor_loss.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
