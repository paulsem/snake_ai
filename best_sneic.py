from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

import random
import numpy as np
import math


class BestAI(object):

    def __init__(self):
        self.input = 8
        self.output = 4

        self.weights = False

        self.y = 0.9
        self.n = 0.0005

        self.reward = 0
        self.final_reward = 0

        self.model = self.makeModel()
        self.model = self.makeModel(weights="cel_mai_smecher.h5")
        self.memory = []

    def makeReward(self, game):
        self.reward -= 0.001
        self.final_reward = 0
        if game.game_over():
            self.reward = 0
            self.final_reward = -1000
        elif game.getEaten():
            self.reward = 0
            self.final_reward = game.getScore() * 1000
        else:
            prev = math.sqrt(
                (game.player.body[1].pos.x - game.food.pos.x) ** 2 + (game.player.body[1].pos.y - game.food.pos.y) ** 2)
            current = math.sqrt(
                (game.player.head.pos.x - game.food.pos.x) ** 2 + (game.player.head.pos.y - game.food.pos.y) ** 2)
            if prev > current:
                self.reward += 0.04
            else:
                self.reward -= 0.04
        return self.final_reward + self.reward

    def makeModel(self, weights=None):
        model = Sequential()
        model.add(Dense(activation='relu', input_dim=self.input, units=100))
        model.add(Dropout(0.5))
        model.add(Dense(activation='softmax', units=self.output))
        model.compile(loss='mse', optimizer=Adam(self.n))

        if weights:
            self.weights = True
            model.load_weights(weights)
        return model

    def makeMemory(self, tmp_memory):
        self.memory.append(tmp_memory[0])

    def trainMemory(self, game, tmp_memory):
        if len(tmp_memory) > 1:
            sample = random.sample(tmp_memory, int(1 / game.getIteration() * len(tmp_memory)))
        else:
            sample = tmp_memory

        for old_states, move, reward, next_state, over in sample:
            tmp_target = reward
            if not over:
                tmp_target = reward + self.y * np.amax(self.model.predict(np.array([next_state]))[0])
            target = self.model.predict(np.array([old_states]))
            target[0][np.argmax(move)] = tmp_target
            self.model.fit(np.array([old_states]), target, epochs=1, verbose=0)
