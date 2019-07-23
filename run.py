import argparse

from kereste import Experiment
from visualize import visualize

def run(path):
    experiment = Experiment(path=path)
    experiment.run(exec_every_epoch=False)
    experiment.execute()
    experiment.save()
    visualize(experiment.conf['results'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the given experiment.')
    parser.add_argument('--experiment', dest='experiment', type=str, required=True, help='Experiment path')
    
    args = parser.parse_args()
    run(args.experiment)