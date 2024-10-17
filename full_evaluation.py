import json
from run_experiments import run
with open('envs.json') as file:
    environment_list = json.load(file)

models = ['rand', 'gpt4', 'gpt4o', 'gemini', 'claude']

for model in models:
    environments = list(environment_list.keys())
    for game in environments:
        print('Running test for: ', game)
        print('\n\n')
        results = run(game, environment_list[game], model, True)
