import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
print("Gym:", gym.__version__)
print("Tensorflow:", tf.__version__)


class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=5000)     # Buffer
		self.gamma = 0.95                    # discount rate
		self.epsilon = 1.0                   # zvědavost
		self.epsilon_min = 0.01              # min zvědavost
		self.epsilon_decay = 0.999           # jak moc rychle se má zmenšovat "zvědavost"
		self.learning_rate = 0.001
		self.model = self._build_model()

		
	def _build_model(self):
		"""
			Deep Q Neural Network  
		"""
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
		return model

	
	def remember(self, state, action, reward, next_state, done):
		"""
			Buffer k zapamatování historie 
		"""
		self.memory.append((state, action, reward, next_state, done))

		
	def act(self, state):
		"""
			Na začátku, když není síť trénovaná mohla by se už na začátku zacyklit
			ve špatných rozhodnutích. Proto je na začátku důležité dělat 
			rozhodnutí o akcích náhodně. Proto je tu přidaná možnost náhodného
			rozhodnutí, jejíž pravděpodobnost se s počtem episod zmenšuje. 
		"""
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	
	def replay(self, batch_size):
		"""
			Trénovací funkce, která přehrává a trénuje již provdené scénáře z Bufferu
		"""
		minibatch = random.sample(self.memory, batch_size)
		states, targets_f = [], []
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target 
			
			states.append(state[0])
			targets_f.append(target_f[0])
		history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
		# Keeping track of loss
		loss = history.history['loss'][0]
		if self.epsilon > self.epsilon_min:   # zmenšení zvědavosti
			self.epsilon *= self.epsilon_decay
		return loss

	
	def load(self, name):
		self.model = keras.models.load_model(name)

		
	def save(self, name):
		self.model.save(name)


def train():
	EPISODES = 400
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)
	done = False
	batch_size = 50

	for e in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			# env.render()
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			#reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
					  .format(e+1, EPISODES, time+1, agent.epsilon))
				break
			if len(agent.memory) > batch_size:
				loss = agent.replay(batch_size)
		if (e+1) % 10 == 0:
			agent.save(f"dqn-{e}.h5") # nejlepší model je uložen pod 'model_24x24.h5'
	return agent, loss


def test_it(agent, env, num):
	for i in range(num):
		state = env.reset()
		total_reward = 0
		done = False
		state = state.reshape(1,state_size)
		time = 0
		agent.epsilon = 0.0
		while not done:
			time += 1
			action = agent.act(state)
			next_state, reward, done, info = env.step(action)
			#env.render()
			next_state = next_state.reshape(1, state_size)
			total_reward += reward
			state = next_state
		   
		print(f"Game with reward {total_reward}")



def env_render(agent, env):
	state = env.reset()
	total_reward = 0
	done = False
	state = state.reshape(1,state_size)
	time = 0
	agent.epsilon = 0.3
	while not done:
		time += 1
		action = agent.act(state)
		next_state, reward, done, info = env.step(action)
		env.render()
		next_state = next_state.reshape(1, state_size)
		total_reward += reward
		state = next_state

	print(f"Game with reward {total_reward}")



env = gym.make('CartPole-v1')
#env = gym.wrappers.Monitor(env, 'not trained', force=True)
#env.env.env.theta_threashold_radians = 3.14/2
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.load("model_24x24.h5")


test_it(agent, env, 5)
env_render(agent, env)
