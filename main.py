import gym
from agent import Agent


if __name__ == "__main__":
    env = gym.make("CarRacing-v1")
    agent = Agent(env)
    agent.train()
    # agent.play_with_model('nasz3.pth')
    env.close()




