from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
import math


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        # self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def makeState(self, game):
        state, direction = game.getGameState(), game.getDirection()

        ok = False
        left, right, up, down = 0, 0, 0, 0
        x = 20
        for body in state["snake_body_pos"]:
            if ok:
                for i in range(1, x):
                    # print(state["snake_head_x"] + i)
                    if not direction == "left" and (
                            state["snake_head_x"] + i == int(body[0]) or state["snake_head_x"] + i >= 600):
                        right = 1
                    if not direction == "right" and (
                            state["snake_head_x"] - i == int(body[0]) or state["snake_head_x"] - i <= 0):
                        left = 1
                    # print(state["snake_head_y"] + i)
                    if not direction == "up" and (
                            state["snake_head_y"] + i == int(body[1]) or state["snake_head_y"] + i >= 600):
                        down = 1
                    if not direction == "down" and (
                            state["snake_head_y"] - i == int(body[1]) or state["snake_head_y"] - i <= 0):
                        up = 1

            else:
                ok = True
        ai_state = [left, right, up, down,
                    state["food_x"] < state["snake_head_x"], state["food_x"] > state["snake_head_x"],
                    state["food_y"] < state["snake_head_y"], state["food_y"] > state["snake_head_y"]]
        for i in range(4, len(ai_state)):
            if ai_state[i]:
                ai_state[i] = 1
            else:
                ai_state[i] = 0
        return np.asarray(ai_state)

    def set_reward(self, game):
        self.reward = -math.sqrt((game.player.head.pos.x - game.food.pos.x) ** 2 + (game.player.head.pos.y - game.food.pos.y) ** 2)
        if game.game_over():
            self.reward = 0
            return -1000
        if game.getEaten():
            self.reward = 0
            return 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=8))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=4, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 8)))[0])
        target_f = self.model.predict(state.reshape((1, 8)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 8)), target_f, epochs=1, verbose=0)
