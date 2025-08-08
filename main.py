from Td3Agent import*
from env import *
import time  # Added for time.sleep

# Define all hyperparameters here
# Environment and training parameters
STATE_DIM = 3  # [cos(theta), sin(theta), theta_dot]
ACTION_DIM = 1
MAX_ACTION = 1


START_EPISODES = 0
MAX_EPISODES = START_EPISODES + 100  # Total episodes
MAX_STEPS = 200  # Timesteps per episode
BATCH_SIZE = 128
TRAININGCYCLES = 300  # Number of training iterations per step

# TD3-specific hyperparameters
DISCOUNT = 0.7
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
EXPL_NOISE_SIGMA = 0.1
NOISE_RESAMPLE_INTERVAL = 5

# Actor and Critic learning rates (currently hardcoded in TD3, but could be added as params if needed)
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

def calculate_reward(angle, angle_dot, action):
    # Reward calculation (same as Gym Pendulum-v1)
    angle = abs(angle) - np.pi

    cost = ((angle) ** 2 + 0.01 * angle_dot ** 2 + 0.01 * (action ** 2))
    return -cost.item()  

# Initialize policy with all hyperparameters
policy = TD3(
    STATE_DIM, 
    ACTION_DIM, 
    MAX_ACTION, 
    discount=DISCOUNT, 
    tau=TAU, 
    policy_noise=POLICY_NOISE, 
    noise_clip=NOISE_CLIP, 
    policy_freq=POLICY_FREQ, 
    expl_noise_sigma=EXPL_NOISE_SIGMA, 
    noise_resample_interval=NOISE_RESAMPLE_INTERVAL
)

replay_buffer = ReplayBuffer(STATE_DIM, ACTION_DIM)

evaluations = []

env = env()

episode_reward = 0
episode_timesteps = 0
episode_num = 0
total_timesteps = 0

states = np.zeros((MAX_STEPS, STATE_DIM), dtype=float)  # 2D: jeder Schritt ein Zustand
angles = np.zeros((MAX_STEPS, 1), dtype=float)  # 2D: Winkel pro Schritt
actions = np.zeros((MAX_STEPS, 1), dtype=float)   # 2D: eine Aktion pro Schritt
dones = np.zeros((MAX_STEPS, 1), dtype=bool)    # 2D: done-Flags
step = 0

def workUp_episode():
    global episode_reward, step

    for i in range(step):
        reward = calculate_reward(angles[i], states[i][2], actions[i])
        episode_reward += reward
        if i == step - 1:
            replay_buffer.add(states[i], actions[i], np.zeros(STATE_DIM), reward, dones[i])
        else:
            replay_buffer.add(states[i], actions[i], states[i + 1], reward, dones[i])

    evaluate_episode()

    states[:] = 0
    actions[:] = 0
    dones[:] = 0
    step = 0

def train_agent():
    # Train agent after collecting sufficient data
    if episode_num >= START_EPISODES:
        print(f"Training agent with {replay_buffer.size} samples...")
        for _ in range(TRAININGCYCLES):  # Train for TRAININGCYCLES iterations
            policy.train(replay_buffer, BATCH_SIZE)

def evaluate_episode():
    global episode_reward, episode_timesteps, episode_num, total_timesteps
    print(f"Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1 


print("Starting training...")
env.start_episode()
while episode_num < MAX_EPISODES:

    episode_timesteps += 1
    total_timesteps += 1

    current_step, state, angle, done = env.get_state()

    # Select action randomly or according to policy
    if episode_num < START_EPISODES:
        action = np.random.uniform(-MAX_ACTION, MAX_ACTION, ACTION_DIM)
    else:
        action = policy.select_action(np.array(state))

    # Perform action
    env.send_action(action.item())
    done = done or episode_timesteps >= MAX_STEPS

    states[step] = state
    angles[step] = angle
    actions[step] = action
    dones[step] = done

    step += 1

    if done or episode_timesteps >= MAX_STEPS:
        workUp_episode()
        train_agent()

        if episode_num < MAX_EPISODES:         
            time.sleep(10)
            env.start_episode()

env.close()
# Save the trained actor model
torch.save(policy.actor.state_dict(), "td3_pendulum_actor.pth")