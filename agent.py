import random
from model import DQN
import torch.nn as nn
import torch.optim as optim
from collections import deque

from helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:

    def __init__(self, env):
        self.total_steps = 0
        self.env = env
        self.memory = deque(maxlen=MAX_MEMORY)
        self.policy_net = DQN(*MAP_SIZE, ACTIONS_NUMBER, ACTION_SKIP).to(device)
        self.target_net = DQN(*MAP_SIZE, ACTIONS_NUMBER, ACTION_SKIP).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()  # mean square error loss
        self.n_games = 0

    def select_action(self, state, only_model=False):
        sample = random.random()
        eps_threshold = get_epsilon(self.total_steps)
        if eps_threshold < EPS_END:
            eps_threshold = EPS_END
        self.total_steps += 1
        if sample > eps_threshold or only_model is True:
            with torch.no_grad():
                actionQVals = self.policy_net(state)
                selected_action_index = actionQVals.max(1)[1].view(1, 1)
                return selected_action_index
        else:
            random_select = torch.tensor([[random.randrange(ACTIONS_NUMBER)]], device=device, dtype=torch.long)
            return random_select

    def get_res_state(self, action, from_model=False):
        accumulated_reward = 0
        accumulated_screen = torch.Tensor().to(device)

        for i in range(ACTION_SKIP):
            if from_model or self.n_games % RENDER_EVERY == 0:
                self.env.render()
            s_observation, s_reward, s_done, _ = self.env.step(action)
            accumulated_reward += s_reward
            screen = get_screen(s_observation)
            accumulated_screen = torch.cat((accumulated_screen, screen), dim=1)

        return accumulated_screen, accumulated_reward, s_done

    # perform one step of model optimisation, select data from replay
    # memory and compute old and updated Q values
    def update_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch_sample)

        # split data batches to be evaluated by models
        next_state_batch = torch.cat(next_states)
        state_batch = torch.cat(states)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        # observe current state values of network
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # compute next state Q values for Q learning computation
        next_state_max_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_max_values * GAMMA) + reward_batch
        # Compute loss
        loss = self.criterion(state_action_values.squeeze(1), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.n_games % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss)

    def train(self):
        score_averages = []
        all_scores = []
        reward_last10 = []

        # perform model training
        for i_episode in range(TRAINING_EPISODES):
            self.n_games += 1
            score = 0
            neg_cnt = 0

            # Initialize the environment and state
            print("Starting episode: ", i_episode)
            self.env.reset()
            wait_for_zoom(self.env)
            state, _, _ = self.get_res_state([0, 0, 0])

            # perform steps in episode, episode ends when ENV returns done == true
            for step in range(STEPS_IN_EPISODE):
                action_index = self.select_action(state)
                action = ACTION_SPACE[action_index]
                next_state, reward, done = self.get_res_state(action)

                score += reward

                neg_cnt = neg_cnt + 1 if reward < 0 else 0

                # transform reward to required format
                reward = float(reward)
                reward = torch.tensor([reward], device=device)

                # Observe new state
                if not done:
                    self.memory.append((state, action_index, reward, next_state, done))
                    state = next_state

                # Perform one step of the optimization (on the target network)
                if self.n_games % UPDATE_FREQ == 0:
                    self.update_model()

                if done or neg_cnt == MAX_NEGATIVE_REWARDS:
                    message = f"Done finished or timeout - steps {step} - score {score}" if neg_cnt < MAX_NEGATIVE_REWARDS \
                        else f"Done, max_negative_steps in row - steps {step} - score {score}"
                    print(message)
                    break

            all_scores.append(score)

            # save model
            if i_episode % SAVE_MODEL == 0:
                self.target_net.save(f'model{i_episode}.pth')

            plot(i_episode, all_scores, reward_last10, score_averages)

    def play_with_model(self, file_name):

        self.policy_net.load(file_name)
        self.policy_net.eval()

        self.env.reset()
        wait_for_zoom(self.env)
        state, _, _ = self.get_res_state([0, 0, 0])
        for i in range(1000):
            action_index = self.select_action(state, True)
            next_state, reward, done = self.get_res_state(ACTION_SPACE[action_index])
            state = next_state
            # if done:
            #     break
