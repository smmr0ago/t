# v3
# model data : numpy -> torch

# usage
# when train : python this_file??.py  --train
# when eval : python this_file??.py --test

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from skimage.color import rgb2gray
from skimage.transform import resize

import gymnasium as gym


def main(args):
    # is_train = args['train']
    # is_test = args['test']
    # is_render = args['human']


    global_step = 0
    score_avg = 0
    score_max = 0

    # 불필요한 행동을 없애주기 위한 딕셔너리 선언
    action_dict = {0:0, 1:1, 2:2, 3:3}      # action_dict = {0:1, 1:2, 2:3, 3:3}

    num_episode = 50000


    if args['train']:
        # 환경과 DQN 에이전트 생성
        env = gym.make('BreakoutDeterministic-v4')
        agent = DQNAgent(state_size=(4, 84, 84), action_size=4)

        if os.path.isfile('breakout.pth'):
            agent.model.load_state_dict(torch.load('breakout.pth'))
            agent.update_target_model()
            print('model loaded - breakout.pth')

        for episode in range(1, num_episode+1):
            done = False
            dead = False
            step, score, start_life = 0, 0, 5
            #reward_t, step_t = 0, 0

            observe, _ = env.reset()

            # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
            # for _ in range(random.randint(1, agent.no_op_steps)):
            #     observe, _, _, _, _ = env.step(1)

            # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
            state = pre_processing(observe)     # (84, 84)
            history = np.stack((state, state, state, state), axis=0)    # (4,84,84)
            history = np.reshape([history], (1, 4, 84, 84))

            while not done:
                global_step += 1
                step += 1
                #reward_t = 0

                # 바로 전 history를 입력으로 받아 행동을 선택
                action = agent.get_action(history)
                # 1: 정지, 2: 왼쪽, 3: 오른쪽 
                # 0:NOOP 1:FIRE 2:RIGHT 3:LEFT

                real_action = action_dict[action]
                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action, real_action, dead = 0, 1, False

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                observe, reward, terminated, truncated, info = env.step(real_action)
                done = truncated or terminated
                
                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 1, 84, 84))
                next_history = np.append(next_state, history[:, :3, :, :], axis=1)

                # agent.avg_q_max += np.amax(agent.model( torch.from_numpy(np.float32(history / 255.)) ).detach().numpy())
                # q_max = torch.max(agent.model(torch.from_numpy(np.float32(history / 255.))))
                # agent.avg_q_max += q_max

                #reward_t = 0.1
                if start_life > info['lives']:
                    dead = True
                    start_life = info['lives']

                score += reward
                reward = np.clip(reward, -1., 1.)
                #reward = reward + reward_t

                # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
                agent.append_sample(history, action, reward, next_history, done)

                # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
                if len(agent.memory) >= agent.train_start:
                    agent.train_model()
                    # # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()
                        print('target_model updated')
                    # 모델 저장
                    if global_step % (agent.update_target_rate * 10) == 0:
                        torch.save(agent.model.state_dict(), "breakout.pth")
                        print('model saved - breakout.pth')

                # if dead:
                #     history = np.stack((next_state, next_state,
                #                         next_state, next_state), axis=0)
                #     history = np.reshape([history], (1, 4, 84, 84))
                # else:
                history = next_history

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    log = "episode: {:5d} | ".format(episode)
                    #log += "reward: {:4.1f} | ".format(reward)
                    log += "score: {:4.1f} | ".format(score)
                    log += "score max : {:4.1f} | ".format(score_max)
                    log += "score avg: {:4.1f} | ".format(score_avg)
                    log += "memory length: {:5d} | ".format(len(agent.memory))
                    log += "epsilon: {:.3f} | ".format(agent.epsilon)
                    #log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                    log += "avg loss : {:3.2f} | ".format(agent.avg_loss / float(step))
                    log += "global_step: {:5d} | ".format(global_step)
                    print(log)
                    
                    agent.avg_q_max, agent.avg_loss = 0, 0



            # 1000 에피소드마다 모델 저장
            if episode % 1000 == 0:
                torch.save(agent.model.state_dict(), "breakout.pth")
                print('model saved - breakout.pth')


    if args['human']:
        # 환경과 DQN 에이전트 생성
        env = gym.make('BreakoutDeterministic-v4')
        agent = DQNAgent(state_size=(4, 84, 84), action_size=4)

        if os.path.isfile('breakout.pth'):
            agent.model.load_state_dict(torch.load('breakout.pth'))
            agent.update_target_model()
            print('model loaded - breakout.pth')

        episodes_data = []
        episode_data = []
        with open(file='breakout_data.pkl', mode='rb') as f:
            episodes_data = pickle.load(f)

        for episode_data in episodes_data:
            done = False
            dead = False
            step, score, start_life = 0, 0, 5
            #reward_t, step_t = 0, 0

            observe, _ = env.reset()

            
            # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
            # for _ in range(random.randint(1, agent.no_op_steps)):
            #     observe, _, _, _, _ = env.step(1)

            # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
            state = pre_processing(observe)     # (84, 84)
            history = np.stack((state, state, state, state), axis=0)    # (4,84,84)
            history = np.reshape([history], (1, 4, 84, 84))

            while not done:
                global_step += 1
                step += 1
                #reward_t = 0

                state, action, reward, next_state, done = episode_data



                # # 바로 전 history를 입력으로 받아 행동을 선택
                # action = agent.get_action(history)
                # # 1: 정지, 2: 왼쪽, 3: 오른쪽 
                # # 0:NOOP 1:FIRE 2:RIGHT 3:LEFT

                # real_action = action_dict[action]
                # # 죽었을 때 시작하기 위해 발사 행동을 함
                # if dead:
                #     action, real_action, dead = 0, 1, False

                # # 선택한 행동으로 환경에서 한 타임스텝 진행
                # observe, reward, terminated, truncated, info = env.step(real_action)
                # done = truncated or terminated

                

                
                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(next_state)
                next_state = np.reshape([next_state], (1, 1, 84, 84))
                next_history = np.append(next_state, history[:, :3, :, :], axis=1)

                #reward_t = 0.1
                # if start_life > info['lives']:
                #     dead = True
                #     start_life = info['lives']

                score += reward
                reward = np.clip(reward, -1., 1.)
                #reward = reward + reward_t

                # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
                agent.append_sample(history, action, reward, next_history, done)

                # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
                if len(agent.memory) >= agent.train_start:
                    agent.train_model()
                    # # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()
                        print('target_model updated')
                    # 모델 저장
                    if global_step % (agent.update_target_rate * 10) == 0:
                        torch.save(agent.model.state_dict(), "breakout.pth")
                        print('model saved - breakout.pth')

                # if dead:
                #     history = np.stack((next_state, next_state,
                #                         next_state, next_state), axis=0)
                #     history = np.reshape([history], (1, 4, 84, 84))
                # else:
                history = next_history

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    log = "episode: {:5d} | ".format(episode)
                    log += "reward: {:4.1f} | ".format(reward)
                    log += "score: {:4.1f} | ".format(score)
                    #log += "score max : {:4.1f} | ".format(score_max)
                    #log += "score avg: {:4.1f} | ".format(score_avg)
                    log += "memory length: {:5d} | ".format(len(agent.memory))
                    log += "epsilon: {:.3f} | ".format(agent.epsilon)
                    #log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                    log += "avg loss : {:3.2f} | ".format(agent.avg_loss / float(step))
                    log += "global_step: {:5d} | ".format(global_step)
                    print(log)
                    
                    agent.avg_q_max, agent.avg_loss = 0, 0



            # 1000 에피소드마다 모델 저장
            if episode % 1000 == 0:
                torch.save(agent.model.state_dict(), "breakout.pth")
                print('model saved - breakout.pth')


    if args['test']:
        env = gym.make('BreakoutDeterministic-v4', render_mode="human")
        agent = DQNAgent(state_size=(4, 84, 84), action_size=4)

        agent.model.load_state_dict(torch.load('breakout.pth'))
        num_episode = 2

        for episode in range(num_episode):
            done = False
            dead = False

            step, score, start_life = 0, 0, 5
            # env 초기화
            observe, _ = env.reset()
            observe, _, _, _, _ = env.step(1)

            # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
            state = pre_processing(observe)     # (84, 84)
            history = np.stack((state, state, state, state), axis=0)    # (4,84,84)
            history = np.reshape([history], (1, 4, 84, 84))

            while not done:
                global_step += 1
                step += 1

                # 바로 전 history를 입력으로 받아 행동을 선택
                agent.epsilon = -1.
                action = agent.get_action(history)
                # 1: 정지, 2: 왼쪽, 3: 오른쪽 
                # 0:NOOP 1:FIRE 2:RIGHT 3:LEFT
                
                real_action = action_dict[action]

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action, real_action, dead = 0, 1, False

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                observe, reward, terminated, truncated, info = env.step(real_action)
                done = truncated or terminated
                
                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 1, 84, 84))
                next_history = np.append(next_state, history[:, :3, :, :], axis=1)

                # agent.avg_q_max += np.amax(agent.model( torch.from_numpy(np.float32(history / 255.)) ).detach().numpy())
                # q_max = torch.max(agent.model(torch.from_numpy(np.float32(history / 255.))))
                # agent.avg_q_max += q_max
                
                if start_life > info['lives']:
                    dead = True
                    start_life = info['lives']

                # score += reward
                # reward = np.clip(reward, -1., 1.)
                
                # if dead:
                #     history = np.stack((next_state, next_state, next_state, next_state), axis=0)
                #     history = np.reshape([history], (1, 4, 84, 84))
                # else:
                history = next_history

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    log = "episode: {:5d} | ".format(episode)
                    log += "score: {:4.1f} | ".format(score)
                    #log += "score max : {:4.1f} | ".format(score_max)
                    #log += "score avg: {:4.1f} | ".format(score_avg)
                    log += "memory length: {:5d} | ".format(len(agent.memory))
                    log += "epsilon: {:.3f} | ".format(agent.epsilon)
                    #log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                    log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
                    print(log)

                    agent.avg_q_max, agent.avg_loss = 0, 0

        env.close()
        # exit()







# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):  # 1*6*210*160
        super(DQN, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        self.conv1 = nn.Conv2d(channels, 32, (8, 8), stride=(4, 4), padding=1)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2), padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        
        self.l1 = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU()
        )

        conv_output_size = self.conv_output_dim()
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_output_size, 512)
        self.fc_out = nn.Linear(512, output_dim)

        self.l2 = nn.Sequential(
            self.fc,
            nn.ReLU(),
            self.fc_out
        )

    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))
        
    def forward(self, x):
        x = self.l1(x)
        x = self.flatten(x)
        q = self.l2(x)
        return q



# 브레이크아웃 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = self.epsilon, 0.05
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 리플레이 메모리, 최대 크기 100,000
        self.memory = deque(maxlen=100000)
        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 30

        # 모델과 타깃 모델 생성
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.eval()

        #self.max_norm = 10.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # 타깃 모델 초기화
        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0

        #self.writer = tf.summary.create_file_writer('summary/breakout_dqn')
        #self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(torch.tensor(history).to(self.device))
            return q_value.argmax().item()      # np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch = random.sample(self.memory, self.batch_size) #batch 32 (1,4,84,84),1,1,(1,4,84,84),1

        history = np.array([sample[0][0] / 255. for sample in batch], dtype=np.float32)     # sample[0] (1,4,84,84) (32,4,84,84) 0.0~0.6
        actions = np.array([sample[1] for sample in batch])                                 # (32,)
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] / 255. for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        history = torch.tensor(history).to(self.device) #[32,4,84,84]
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)    #torch.Size([32,1])
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        next_history = torch.tensor(next_history).to(self.device)
        dones = torch.tensor(dones).float().unsqueeze(1).to(self.device)

        # # 현재 상태에 대한 모델의 큐함수
        # predicts = torch.gather(self.model(torch.tensor(history)), 1, torch.tensor(actions))

        # # 다음 상태에 대한 모델의 큐함수
        # # 벨만 최적 방정식을 구성하기 위한 타깃과 큐함수의 최대 값 계산
        # # max_q = np.amax(target_predicts, axis=1)  
          
        # # next_q_values = self.model(torch.tensor(next_history))
        # # max_next_q = torch.max(next_q_values)
        # max_next_q = self.target_model(torch.tensor(next_history)).detach().max(1)[0].unsqueeze(1)
        # targets = torch.tensor(rewards) + (1 - dones) * self.discount_factor * max_next_q
        # # targets = targets.unsqueeze(-1)
        # # targets = targets.type(torch.float32)

        predicts = torch.gather(self.model(history), 1, actions)
        max_next_q = self.target_model(next_history).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q

        loss = nn.SmoothL1Loss()(predicts, targets).to(self.device)
        #self.avg_loss += loss.detach().numpy()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
    


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '--t', action='store_true')
    parser.add_argument('--test', '--s', action='store_true')
    parser.add_argument('--human', '--h', action='store_true')

    main(vars(parser.parse_args()))
