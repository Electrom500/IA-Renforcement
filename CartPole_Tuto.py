### Tuto from anyscale
import gymnasium
env = gymnasium.make('CartPole-v1')

#permet de swicth entre les différents scripts
#valeurs de run = "alea","DQL"
run = "alea"


def cartPole_aleatoire():
    # returns an initial observation
    env.reset()

    for i in range(20):
        # env.action_space.sample() produces either 0 (left) or 1 (right).
        observation, reward, done, truncated, info = env.step(env.action_space.sample())

        print("step", i, observation, reward, done, truncated, info)

    return

def cartPole_DQL():
    return



match run:
    case "alea":
        cartPole_aleatoire()
    case "DQL":
        cartPole_DQL()
    case _:
        print("Wrong choice in run value")

env.close()