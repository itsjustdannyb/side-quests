import gymnasium as gym
from collections import defaultdict, Counter
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
ALPHA = 0.2
GAMMA = 0.9
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        # in q-learning, tracking just our value table will make our memory footprint smaller
        self.values = defaultdict(float) 

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        self.state = self.env.reset()[0] if terminated or truncated else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action 

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v # Bellman Approximation
        old_v = self.values[(s,a)]
        self.values[(s,a)] = old_v * (1-ALPHA) + new_v * ALPHA # update approximations using blending technique

    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()

        while True:
            # get the best action to perform in this state
            _, best_action = self.best_value_and_action(state)
            # pass the best action to the env
            new_state, reward, terminated, truncated, _ = env.step(best_action)
            total_reward += reward

            if terminated or truncated:
                break
            state = new_state
        return total_reward

        
if __name__ == "__main__":
    test_env = gym.make(ENV_NAME, render_mode='rgb_array')
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= float(TEST_EPISODES)
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"Best Reward Updated: {best_reward:.3f} -> {reward:.3f} ")
            best_reward = reward
        if reward > 0.8:
            print(f"Solved in {iter_no} iterations!")
            break    
    
    writer.close()
