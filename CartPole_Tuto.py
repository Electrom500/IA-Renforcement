import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import ray
from ray import tune
from ray.rllib.algorithms import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

os.makedirs('./videos', exist_ok=True)


#def the environnement
env = gym.make('CartPole-v1', render_mode="rgb_array")

# add the Recording option to the environnement
#env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda episode_id: True)

#allow to switch btw diff scripts
#valeurs de run = "alea","DQL"
run = "DQL"

def cartPole_aleatoire():
    observation, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("step", i, observation, reward, terminated, truncated, info)

        #if terminated or truncated:
         #   observation, info = env.reset()
          #  break


def cartPole_DQL():
    global env
    # Entraînement de l'agent PPO
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .training(
            model={"fcnet_hiddens": [32], "fcnet_activation": "linear"}
        )
    )

    stop = {"env_runners/episode_return_mean": 50}  # ajustable

    result = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_at_end=True,
    )

    checkpoint_path = result.get_best_checkpoint(
        trial=result.get_best_trial("episode_return_mean", mode="max"),
        metric="episode_return_mean",
        mode="max"
    )

    algo = PPO(config=config)
    algo.restore(checkpoint_path)

    # Préparer l’environnement pour la vidéo
    os.makedirs("./videos", exist_ok=True)
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)

    # Lancer une session avec la vidéo
    obs, _ = env.reset()
    for _ in range(500):
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()



match run:
    case "alea":
        cartPole_aleatoire()
    case "DQL":
        cartPole_DQL()
    case _:
        print("Wrong choice in run value")


ray.shutdown()
env.close()
