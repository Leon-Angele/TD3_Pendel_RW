from sympy import re
from Td3Agent import*
from env import *

def calculate_reward(angle, angle_dot, action):
    # Reward calculation (same as Gym Pendulum-v1)
    angle = abs(angle)- np.pi

    cost = ((angle) ** 2 + 0.1 * angle_dot ** 2 + 0.001 * (action ** 2))
    return -cost.item()  

# Parameters
START_EPISODES = 5
MAX_EPISODES = START_EPISODES + 50  # Total episodes
MAX_STEPS = 800  # Timesteps per episode
EVAL_FREQ = 10  # Evaluate every 10 episodes
EXPL_NOISE = 0.2
BATCH_SIZE = 256
TRAININGCYCLES = 300  # Number of training iterations per step

state_dim = 3  # [cos(theta), sin(theta), theta_dot]
action_dim = 1 
max_action = 1

policy = TD3(state_dim, action_dim, max_action)

replay_buffer = ReplayBuffer(state_dim, action_dim)


evaluations = []

env = env()

episode_reward = 0
episode_timesteps = 0
episode_num = 0
total_timesteps = 0

states = np.zeros((MAX_STEPS, state_dim), dtype=float)  # 2D: jeder Schritt ein Zustand
angles = np.zeros((MAX_STEPS, 1), dtype=float)  # 2D: Winkel pro Schritt
actions = np.zeros((MAX_STEPS, 1), dtype=float)   # 2D: eine Aktion pro Schritt
dones = np.zeros((MAX_STEPS, 1), dtype=bool)    # 2D: done-Flags
step = 0


def workUp_episode():
    global episode_reward

    for i in range(len(states)):
        reward = calculate_reward(angles[i], states[i][2], actions[i])
        episode_reward += reward
        if i == len(states)-1:
            replay_buffer.add(states[i], actions[i], np.zeros(3) , reward, dones[i])
        else:
            replay_buffer.add(states[i], actions[i], states[i + 1], reward, dones[i])

    evaluate_episode()

    states[:] = 0
    actions[:] = 0
    dones[:] = 0

def train_agent():
    # Train agent after collecting sufficient data
    if episode_num >= START_EPISODES:
        print(f"Training agent with {replay_buffer.size} samples...")
        for _ in range(TRAININGCYCLES):  # Train for 100 iterations
            policy.train(replay_buffer, BATCH_SIZE)



def evaluate_episode():
    global episode_reward, episode_timesteps, episode_num, total_timesteps
    print(f"Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1 

    '''
    # Evaluate episode
    if episode_num % EVAL_FREQ == 0:
        print("---------------------------------------")
        evaluations.append(eval_policy(policy, env, max_steps=MAX_STEPS))
        print("---------------------------------------")
    '''



env.start_episode()

while episode_num < MAX_EPISODES:

    episode_timesteps += 1
    total_timesteps += 1

    step, state, angle ,done = env.get_state()


    # Select action randomly or according to policy (always scalar)
    if total_timesteps < START_EPISODES:
        action = float(np.random.uniform(-max_action, max_action))
    else:
        action = float(
            (policy.select_action(np.array(state))
            + np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
            ).clip(-max_action, max_action)[0]
        )

    # Perform action
    env.send_action(action)
    done = done or episode_timesteps >= MAX_STEPS

    states[step] = state
    angles[step] = angle
    actions[step] = action
    dones[step] = done

    if done or episode_timesteps >= MAX_STEPS:
        
        workUp_episode()
        train_agent()

        if episode_num < MAX_EPISODES:         
            time.sleep(10)
            env.start_episode()



env.close()
# Save the trained actor model
torch.save(policy.actor.state_dict(), "td3_pendulum_actor.pth")








