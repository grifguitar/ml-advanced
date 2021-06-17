import gym
import tensorflow as tf
import collections as cll
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import PIL


class Agent:
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.huber,
                      metrics=['accuracy'])
        return model

    def __init__(self, env, input_shape, max_total_step_cnt, gamma=0.6, max_epsilon=0.3, min_epsilon=0.01):
        self.env = env

        self.input_shape = input_shape
        self.action_size = env.action_space.n

        self.buffer = cll.deque(maxlen=100000)

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


def draw(env, i_episode, i_step, state, reward, info, eps):
    # env.render()
    # plt.imshow(np.reshape(state, (160, 480)))
    # plt.colorbar()
    # plt.show()
    print(i_episode, i_step, reward, info.get('ale.lives'), '%.4f' % eps)


def merge_frames(frames):
    images = list()
    for frame in frames:
        img = PIL.Image.fromarray(frame).convert(mode='L')
        images.append(img)
    widths, heights = zip(*(img.size for img in images))
    max_height = max(heights)
    total_width = sum(widths)
    new_image = PIL.Image.new('L', (total_width, max_height))
    offset = 0
    for img in images:
        new_image.paste(img, (offset, 0))
        offset += img.size[0]
    return np.array(new_image)


def take_action(env, action, frame_buffer):
    _, reward, done, info = env.step(action)
    img = env.render(mode='rgb_array')
    img = img[34:-16, :, :]
    frame_buffer.append(img)

    new_img = merge_frames(frame_buffer)
    new_img = new_img / 255

    new_img = np.reshape(new_img, new_img.shape + (1,))

    return new_img, reward, done, info


def solve():
    FRAME_CNT = 3
    frame_buffer = cll.deque(maxlen=FRAME_CNT)

    env = gym.make('BreakoutDeterministic-v4')

    LAUNCH_BALL_ACTION = 1
    NEED_LAUNCH_BALL = False

    tmp_st = env.reset()
    for i in range(FRAME_CNT):
        tmp_st, _, _, _ = take_action(env, env.action_space.sample(), frame_buffer)
        plt.imshow(np.reshape(tmp_st, (160, -1)))
        plt.colorbar()
        plt.show()

    INPUT_SHAPE = tmp_st.shape

    print('input_shape: ', INPUT_SHAPE)

    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space, env.action_space.n)
    print('------------------------------')

    episodes_count = 100
    max_steps_count = 1000
    batch_size = 20
    copy_period = 10

    agent = Agent(env, INPUT_SHAPE, episodes_count * max_steps_count)
    agent.q_network.summary()

    total_rewards = list()
    total_qs = list()

    for i_episode in range(episodes_count):

        env.reset()
        state, _, _, info = take_action(env, LAUNCH_BALL_ACTION, frame_buffer)
        state = np.reshape(state, (1,) + INPUT_SHAPE)

        total_qs.append(Agent.get_best_value(agent.q_network, state))

        total_reward = 0

        for i_step in range(max_steps_count):

            # выбираем оптимальное действие из текущего состояния
            action = agent.get_action(state)

            # принудительно запустим мяч, если необходимо
            if NEED_LAUNCH_BALL:
                action = LAUNCH_BALL_ACTION
                print("need to launch the ball!")

            # делаем действие и получаем результаты
            next_state, reward, done, new_info = take_action(env, action, frame_buffer)
            next_state = np.reshape(next_state, (1,) + INPUT_SHAPE)
            total_reward += reward

            # делаем выводы из результатов
            if info.get('ale.lives') != new_info.get('ale.lives'):
                NEED_LAUNCH_BALL = True
            else:
                NEED_LAUNCH_BALL = False

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
            draw(env, i_episode, i_step, state, reward, info, agent.epsilon)

            # переходим в новое состояние
            state = next_state
            info = new_info

        agent.copy_weights()

        total_rewards.append(total_reward)

    plt.title("total_rewards")
    plt.plot(total_rewards)
    plt.show()

    plt.title("total_qs")
    plt.plot(total_qs)
    plt.show()

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
