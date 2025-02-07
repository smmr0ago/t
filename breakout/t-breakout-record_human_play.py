import gymnasium as gym
import numpy as np
import pickle
from pynput import keyboard
import time

k = 0
isEsc = False
def on_press(key):
    global k
    if key == keyboard.Key.up:
        k = 1
    elif key == keyboard.Key.right:
        k = 2
    elif key == keyboard.Key.left:
        k = 3
    elif key == keyboard.Key.esc:
        global isEsc
        isEsc = True
def on_release(key):
    global k
    k = 0
        
listener = keyboard.Listener(on_press=on_press,on_release=on_release)
listener.start()

def get_action():
    return k


# 환경 초기화
env = gym.make("BreakoutDeterministic-v4", render_mode="human")
num_episodes = 10  # 플레이할 에피소드 수

data = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_data = []

    while not done:
        env.render()
        action = get_action()
        next_state, reward, terminated, truncated, _ = env.step(action)
        truncated = isEsc
        done = terminated or truncated
        time.sleep(0.03)
        #episode_data.append((state, action, reward, next_state, done))
        state = next_state

    #data.append(episode_data)

env.close()

# 데이터 저장
#with open('breakout_data.pkl', 'wb') as f:
#    pickle.dump(data, f)

