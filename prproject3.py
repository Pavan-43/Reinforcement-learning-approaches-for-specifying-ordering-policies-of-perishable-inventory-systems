import numpy as np

MAX_STOCK = 100
MAX_DAYS_OLD = 6

class AppleShop:
    def __init__(self, initial_stock=50):
        self.stock = initial_stock
        self.days_old = np.zeros(initial_stock)
    
    def step(self, action):
        reward = 0
        done = False
        
        # Generate random demand between 20 and 40
        demand = np.random.randint(20, 41)  # Adjust the range as needed
        
        if action <= self.stock:
            self.stock -= action
            self.stock += demand
            reward += demand - action
            
            # Make a copy of days_old array before modifying it
            days_old_copy = self.days_old.copy()
            
            days_old_copy[:action] += 1
            days_old_copy[action:] = 0
            
            # Update self.days_old with the modified copy
            self.days_old = days_old_copy
            
            if self.stock >= MAX_STOCK:
                done = True
        else:
            reward -= (action - self.stock) * 2  # Penalty for overstocking
        
        i = 0
        while i < self.stock and i < len(self.days_old):  # Ensure i is within bounds
            if self.days_old[i] >= MAX_DAYS_OLD:
                self.stock -= 1
                self.days_old = np.delete(self.days_old, i)
                self.days_old = np.append(self.days_old, 0)
            else:
                i += 1
        
        return self.get_state(), reward, done, {}


    
    def get_state(self):
        return (self.stock, *self.days_old)
    
    def reset(self):
        self.stock = np.random.randint(30, 70)
        self.days_old = np.zeros(self.stock)
        return self.get_state()

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_prob=1.0, min_exploration_prob=0.1, exploration_decay=0.995):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_space_size, action_space_size))
    
    def select_action(self, state):
        state_idx = int(self.flatten_state(state))
        if np.random.rand() < self.exploration_prob:
            return np.random.randint(self.action_space_size)
        return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, done):
        state_idx = int(self.flatten_state(state))  # Convert state to integer index
        action_idx = int(action)  # Convert action to integer index

        old_value = self.q_table[state_idx, action_idx]
        if done:
            target = reward
        else:
            next_state_idx = int(self.flatten_state(next_state))  # Convert next_state to integer index
            target = reward + self.discount_factor * np.max(self.q_table[next_state_idx])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * target
        self.q_table[state_idx, action_idx] = new_value
        if self.exploration_prob > self.min_exploration_prob:
            self.exploration_prob *= self.exploration_decay
    
    def flatten_state(self, state):
        stock, *days_old = state
        flattened_days_old = np.array(days_old).flatten()
        state_idx = stock * (MAX_DAYS_OLD + 1) + flattened_days_old[0]  # Calculate the flattened index
        return state_idx

# Define the environment and agent
env = AppleShop()
state_space_size = (MAX_STOCK + 1) * (MAX_DAYS_OLD + 1)
action_space_size = MAX_STOCK + 1
agent = QLearningAgent(state_space_size, action_space_size)

# Training loop
total_episodes = 1000
for episode in range(total_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# You can now use the trained Q-learning agent for inventory management tasks.
