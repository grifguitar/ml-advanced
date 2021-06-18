import gym
import tensorflow as tf
import collections as cll
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


class Agent:
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.observation_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='mse')
        return model

    def __init__(self, env, max_total_step_cnt, gamma=0.6, max_epsilon=0.3, min_epsilon=0.01):
        self.env = env

        self.observation_shape = env.observation_space.shape
        self.action_size = env.action_space.n

        self.buffer = cll.deque(maxlen=10000)

        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.stride_length = (self.max_epsilon - self.min_epsilon) / max_total_step_cnt
        self.epsilon = self.max_epsilon

        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.copy_weights()

    def copy_weights(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save_to_buffer(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    @staticmethod
    def get_best_action(network, state):
        predictions = network.predict(state)
        return np.argmax(predictions[0])

    @staticmethod
    def get_best_value(network, state):
        predictions = network.predict(state)
        return np.amax(predictions[0])

    def get_action(self, state):
        self.epsilon -= self.stride_length
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(self.q_network, state)

    def retrain(self, batch_size):
        batch = rnd.sample(self.buffer, batch_size)
        for state, action, reward, next_state, terminated in batch:
            predictions = self.q_network.predict(state)
            if terminated:
                predictions[0][action] = reward
            else:
                predictions[0][action] = reward + self.gamma * Agent.get_best_value(self.target_network, next_state)
            self.q_network.fit(state, predictions, epochs=1, verbose=0)


def draw(env, i_episode, i_step, state, reward, eps):
    img = env.render(mode='rgb_array')
    plt.clf()
    plt.title('frame_{x}_{y}.png'.format_map({'x': i_episode, 'y': i_step}))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig('D:/new_images_car/frame_{x}_{y}.png'.format_map({'x': i_episode, 'y': i_step}))
    # env.render()
    print(i_episode, i_step, state, reward, '%.4f' % eps)


def solve():
    env = gym.make('MountainCar-v0')

    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space, env.action_space.n)
    print('------------------------------')

    episodes_count = 40
    max_steps_count = 200
    batch_size = 20
    copy_period = 10

    agent = Agent(env, episodes_count * max_steps_count)
    agent.q_network.summary()

    total_rewards = list()
    total_qs = list()

    for i_episode in range(episodes_count):

        state = env.reset()
        state = np.reshape(state, (1,) + env.observation_space.shape)

        total_qs.append(Agent.get_best_value(agent.q_network, state))

        total_reward = 0

        for i_step in range(max_steps_count):

            # выбираем оптимальное действие из текущего состояния
            action = agent.get_action(state)

            # делаем действие и получаем результаты
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, (1,) + env.observation_space.shape)
            total_reward += reward

            # модифицируем награду
            reward += 300 * (abs(next_state[0][1]) - abs(state[0][1]))

            # сохраняем опыт
            agent.save_to_buffer(state, action, reward, next_state, done)

            # обучаем q_network на случайной выборке размера batch_size из нашего опыта
            if len(agent.buffer) > batch_size:
                agent.retrain(batch_size)

            # если игра закончилась, выходим
            if done:
                print("Episode finished after {} timesteps".format(i_step + 1), total_reward)
                break

            # копируем веса с интервалом в copy_period шагов
            if i_step % copy_period == 0:
                agent.copy_weights()

            # рисуем текущее состояние среды
            draw(env, i_episode, i_step, state, reward, agent.epsilon)

            # переходим в новое состояние
            state = next_state

        agent.copy_weights()

        total_rewards.append(total_reward)

        plt.clf()
        plt.title("total_rewards")
        plt.plot(total_rewards)
        plt.savefig('D:/new_images_car/total_rewards_{x}.png'.format_map({'x': i_episode}))

        plt.clf()
        plt.title("total_qs")
        plt.plot(total_qs)
        plt.savefig('D:/new_images_car/total_qs_{x}.png'.format_map({'x': i_episode}))

        model = agent.q_network

        # serialize model to JSON
        model_json = model.to_json()
        with open("new_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("new_model.h5")
        print("Saved model to disk")

    env.close()


if __name__ == '__main__':
    solve()
