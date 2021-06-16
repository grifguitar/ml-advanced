import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from deep_q_learning import Agent


def draw(env, i_episode, i_step, state, reward, eps):
    img = env.render(mode='rgb_array')
    plt.title('frame_{x}_{y}.png'.format_map({'x': i_episode, 'y': i_step}))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig('new_images/frame_{x}_{y}.png'.format_map({'x': i_episode, 'y': i_step}))
    # plt.show()
    print(i_episode, i_step, state, reward, eps)


def solve():
    env = gym.make('MountainCar-v0')

    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space, env.action_space.n)
    print('------------------------------')

    episodes_count = 2
    max_steps_count = 200

    agent = Agent(env, episodes_count * max_steps_count)

    # load json and create model
    json_file = open('model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    agent.q_network = tf.keras.models.model_from_json(loaded_model_json)
    agent.target_network = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    agent.q_network.load_weights("model1.h5")
    agent.target_network.load_weights("model1.h5")
    print("Loaded model from disk")

    agent.copy_weights()
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
            action = Agent.get_best_action(agent.q_network, state)

            # делаем действие и получаем результаты
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, (1,) + env.observation_space.shape)
            total_reward += reward

            # модифицируем награду
            reward += 300 * (abs(next_state[0][1]) - abs(state[0][1]))

            # если игра закончилась, выходим
            if done:
                print("Episode finished after {} timesteps".format(i_step + 1), total_reward)
                break

            # рисуем текущее состояние среды
            draw(env, i_episode, i_step, state, reward, agent.epsilon)

            # переходим в новое состояние
            state = next_state

        total_rewards.append(total_reward)

    plt.clf()
    plt.title("total_rewards")
    plt.plot(total_rewards)
    plt.show()

    plt.title("total_qs")
    plt.plot(total_qs)
    plt.show()

    env.close()


if __name__ == '__main__':
    solve()
