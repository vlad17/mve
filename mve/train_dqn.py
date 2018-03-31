import tensorflow.contrib.layers as layers
import deepq, sys
from deepq.simple import run_experiment

if __name__ == '__main__':
    run_experiment(model=deepq.models.mlp([64]), horizon=int(sys.argv[1]))
