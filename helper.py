from const import *
import numpy as np
import torchvision.transforms as trans
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_epsilon(global_counter):
    if global_counter >= EPS_DECAY_STEP:
        return EPS_END
    else:
        # linear decay
        r = 1.0 - global_counter / float(EPS_DECAY_STEP)
        return EPS_END + (EPS_START - EPS_END) * r


def get_screen(screen):
    # create transcormation that transforms imag to grayscale
    transformation = trans.Compose([trans.ToTensor(), trans.ToPILImage(), trans.Grayscale(), trans.ToTensor()])

    # remove negative axis values from tensor
    # (they are just there somehow and this fixes them to start at zero)
    screen = np.flip(screen, axis=0).copy()
    screen = np.flip(screen, axis=0).copy()
    # perform afforementioned transformation
    screen = transformation(screen).to(device)

    # cut track features from image
    track = screen[:, :, :]
    track = track.view(1, 1, *MAP_SIZE)

    return track


def wait_for_zoom(env):
    for x in range(0, 50):
        env.step([0, 0, 0])


def plot(i_episode, all_scores, reward_last10, score_averages):
    score_averages.append(sum(all_scores) / (i_episode + 1))
    if i_episode % 10 == 0 and i_episode >= 10:
        plt.figure()
        reward_last10.append(sum(all_scores[i_episode - 10:i_episode]) / 10)
        plt.plot(reward_last10)
        plt.title("avg reward from last 10 episodes")
        plt.savefig("avg_reward_treningu.png")
        plt.close()

    plt.figure()
    plt.plot(all_scores)
    plt.plot(score_averages)

    plt.title("Training :)")
    plt.legend(["total", "avg"])
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    # plot_handle = plt.show(block=False)
    plt.savefig("wykresik_treningu.png")
    plt.close()
