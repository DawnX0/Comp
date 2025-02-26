import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, save_file="q_table.pkl"):
        self.actions = actions
        self.q_table = {}  # Q-values stored as {state: {action: value}}
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.save_file = save_file  # File to save Q-table
        self.load_q_table()  # Load Q-table if exists

    def get_q_values(self, state):
        """Return Q-values for a given state, initializing if necessary."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        return self.q_table[state]

    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            q_values = self.get_q_values(state)
            return max(q_values, key=q_values.get)  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using the Bellman equation."""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        
        best_next_action = max(next_q_values, key=next_q_values.get)
        q_values[action] += self.alpha * (reward + self.gamma * next_q_values[best_next_action] - q_values[action])
        
        self.epsilon *= self.epsilon_decay  # Decay epsilon over time
        self.save_q_table()  # Save after updating

    def save_q_table(self):
        """Save the Q-table to a file."""
        with open(self.save_file, "wb") as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved.")

    def load_q_table(self):
        """Load the Q-table from a file if it exists."""
        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as f:
                self.q_table = pickle.load(f)
            print("Q-table loaded.")
        else:
            print("No Q-table found, starting fresh.")

    def visualize_q_table(self):
        """Visualize the Q-table as a heatmap."""
        if not self.q_table:
            print("Q-table is empty. Train the model first.")
            return

        states = list(self.q_table.keys())
        actions = self.actions

        q_values_matrix = np.zeros((len(states), len(actions)))

        for i, state in enumerate(states):
            for j, action in enumerate(actions):
                q_values_matrix[i, j] = self.q_table[state].get(action, 0)

        plt.figure(figsize=(10, 6))
        plt.imshow(q_values_matrix, cmap="coolwarm", aspect="auto")

        plt.colorbar(label="Q-value")
        plt.xticks(ticks=np.arange(len(actions)), labels=actions)
        plt.yticks(ticks=np.arange(len(states)), labels=[str(s) for s in states], fontsize=8)
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title("Q-table Heatmap")
        plt.show()