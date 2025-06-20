import json
import random
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from tqdm import tqdm
import matplotlib.pyplot as plt

from optimizer.main import SimulatorEnv
from simulator.builder import get_env, get_requests_constrains
from simulator.environment import Environment
from simulator.model.simulator import Simulator
from simulator.utils.data_generator.generator import InputDataGenerator
from optimizer.settings import ENV_SETTINGS, GENERATOR_SETTINGS


class InfoLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if "" in info:
                self.logger.record(f"custom/missed_requests_num", info["missed_requests_num"])
        return True


def _apply_restrictions_to_selection(selection: list[int], requests_constrains: list[list[int]]) -> None:
    for request_id, truck_id in enumerate(selection):
        if truck_id not in requests_constrains[request_id]:
            selection[request_id] = -1


def get_everything_for_run_selection(
        input_file_path: str = 'output/input.json',
        route_file_path: str = 'output/routes.json'
) -> (Environment, Simulator, list[list[int]]):
    with open(input_file_path, 'r') as f:
        input_data = json.load(f)

    with open(route_file_path, 'r') as f:
        routes_data = json.load(f)

    environment: Environment = get_env(input_data, routes_data)
    simulator = Simulator()
    requests_constrains = get_requests_constrains(environment, False)
    return environment, simulator, requests_constrains


def run_random_selections_on_input(input_file_path: str = 'output/input.json', route_file_path: str = 'output/routes.json'):
    environment, simulator, requests_constrains = get_everything_for_run_selection(input_file_path, route_file_path)

    selections_to_try = [
        [-1] * len(environment.requests),
        *[[i] * len(environment.requests) for i in range(len(environment.trucks))]
    ]

    for _ in range(10000):
        if len(selections_to_try) > 0:
            selection = selections_to_try.pop(0)
        else:
            selection = [random.randint(-1, len(environment.trucks)) for _ in range(len(environment.requests))]

        _apply_restrictions_to_selection(selection, requests_constrains)

        selection = tuple(selection)
        try:
            missed_requests = simulator.run(selection, environment)
            print(len(missed_requests))
        except Exception as e:
            print(f'Exception on selection: {selection}! ', e)
            break
    print('Done')


def model_eval(model_path, env, n_episodes=10):
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=1024,
        verbose=1,
        tensorboard_log="output/logs/tensorboard"
    ).load(model_path)

    metrics = {
        "reward": [],
        "missed_requests_num": [],
        "unfinished_ratio": []
    }
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        infos = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            infos.append(info)

        metrics["reward"].append(episode_reward / n_episodes)
        metrics["missed_requests_num"].append( np.mean([elem["missed_requests_num"] for elem in infos]) )
        metrics["unfinished_ratio"].append( np.mean([elem["unfinished_ratio"] for elem in infos]) )
        print(f"{episode}: reward: {metrics['reward']}, missed_requests_num: {metrics['missed_requests_num']},"
              f" unfinished_ratio: {metrics['unfinished_ratio']}")

    # Средняя метрика
    avg_completed = np.mean(metrics["missed_requests_num"])
    print(f"Среднее невыполненных заявок: {avg_completed:.2f}")

    fig, axs = plt.subplots(nrows=1, ncols=3)

    for ax, key in zip(axs, metrics.keys()):
        x = list(range(len(metrics[key])))
        ax.plot(x, metrics[key], linestyle='-', marker='o')
        ax.set_title(str(key))
        ax.grid(True)
    plt.show()

    return metrics


def main():
    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        min_distance=GENERATOR_SETTINGS.min_distance,
        max_distance=GENERATOR_SETTINGS.max_distance
    )
    env = SimulatorEnv(generator)

    # print(model_eval("output/models/best/best_model_600k.zip", env, 25))

    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=1024,
        clip_range=0.6,
        verbose=0,
        tensorboard_log="output/logs/tensorboard",
        policy_kwargs={
            "net_arch": [128] * 5
        }
    )

    eval_callback = EvalCallback(
        env, best_model_save_path="output/models/best",
        log_path="output/logs/tensorboard", eval_freq=2048,
        deterministic=True, render=False)
    info_logger_callback = InfoLoggerCallback()

    model.learn(
        total_timesteps=ENV_SETTINGS.max_num_of_steps * ENV_SETTINGS.epochs_num,
        progress_bar=True,
        callback=[eval_callback, info_logger_callback]
    )
    model.save(f"output/models/{str(datetime.today())}.zip")


if __name__ == '__main__':
    main()
