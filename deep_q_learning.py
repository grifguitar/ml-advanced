import matplotlib.pyplot as plt
import numpy as np
import gym


class QLearningAgent:
    # Q-Learning агент

    def __init__(self, alpha, epsilon, discount, env):
        self.actions = lambda arg: range(env.action_space.n)
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def set_q_value(self, state, action, value):
        self.q_table[state][action] = value

    def get_best_action(self, state):
        # выбираем лучшее действие, согласно стратегии
        best_action = None
        for action in self.actions(state):
            if best_action is None:
                best_action = action
            elif self.get_q_value(state, action) > self.get_q_value(state, best_action):
                best_action = action
        return best_action

    def get_max_q_value(self, state):
        # возвращаем q_value для лушего действия
        return self.get_q_value(state, self.get_best_action(state))

    def get_action(self, state):
        # выбираем действие согласно epsilon-стратегии
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions(state), size=1)[0]
        else:
            action = self.get_best_action(state)
        return action

    def update(self, state, action, next_state, reward):
        # выполняем обновление q_value
        t = self.alpha * (reward + self.discount * self.get_max_q_value(next_state) - self.get_q_value(state, action))
        new_q_value = self.get_q_value(state, action) + t
        self.set_q_value(state, action, new_q_value)


def play_and_train(env, agent, max_steps=10 ** 4, is_only_play=False):
    # запускает игру,
    # используя стратегию agent.get_action(),
    # выполняет обновление agent.update()
    # и возвращает общее вознаграждение

    total_reward = 0.0
    state = env.reset()

    for t in range(max_steps):
        # выбираем действие
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        # выполняем обновление стратегии
        if not is_only_play:
            agent.update(state, action, next_state, reward)

        state = next_state
        total_reward += reward
        if done:
            break

        if is_only_play:
            env.render()

    return total_reward


def solve():
    env = gym.make("Taxi-v3")

    agent = QLearningAgent(alpha=0.5, epsilon=0.1, discount=0.9, env=env)

    rewards = []
    for i in range(5000):
        rewards.append(play_and_train(env, agent))

    plt.plot(rewards)
    plt.show()

    print("total_reward = ", play_and_train(env, agent, is_only_play=True))


if __name__ == '__main__':
    solve()
