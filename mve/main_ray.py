"""
Run a grid of settings for MVE-DDPG on a ray cluster.

Adapted from:
http://ray.readthedocs.io/en/latest/tune.html

For simplicity, resource use is always set to 1 GPU.
Change the number of GPUs per machine to adjust the number of concurrent
tasks allowed on that machine.
"""

import os
import subprocess
import sys
import yaml

import boto3
from botocore.exceptions import ClientError
import numpy as np
import ray
from ray.tune import register_trainable, run_experiments
from ray.tune.median_stopping_rule import MedianStoppingRule

from experiment import setup_experiment_context
from flags import (ArgSpec, Flags, parse_args)
from main_ddpg import train as ddpg_train
from main_ddpg import ALL_DDPG_FLAGS
from main_sac import train as sac_train
from main_sac import ALL_SAC_FLAGS
import reporter


def ngpus():
    """Return number of gpus on current machine"""
    # Do this in a subprocess to avoid creating a session here.
    cmd = 'from tensorflow.python.client import device_lib;'
    cmd += 'local_device_protos = device_lib.list_local_devices();'
    cmd += """print(sum(device.device_type == 'GPU' for device """
    cmd += """in local_device_protos))"""
    prev_env_devices = None
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        prev_env_devices = os.environ['CUDA_VISIBLE_DEVICES']
        del os.environ['CUDA_VISIBLE_DEVICES']
    try:
        out = subprocess.check_output(
            [sys.executable, '-c', cmd], stderr=subprocess.DEVNULL)
    finally:
        if prev_env_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = prev_env_devices
    return int(out.strip())


class TuneFlags(Flags):
    """
    These flags define parameters for the ray tune script.
    """

    @staticmethod
    def _generate_arguments():
        """Adds flags to an argparse parser."""
        yield ArgSpec(
            name='self_host',
            default=False,
            action='store_true',
            help='whether to create a local ray cluster')
        yield ArgSpec(
            name='port',
            default='7001',
            type=str,
            help='default ray port to connect to')
        yield ArgSpec(
            name='server_port',
            default='10000',
            type=str,
            help='default ray port to connect to')
        yield ArgSpec(
            name='median_stop',
            default=int(1e6),
            type=int,
            help='cutoff for median stopping rule, use -1 for none')
        yield ArgSpec(
            name='experiment_name',
            required=True,
            type=str,
            help='experiment name, should be a valid, empty s3 directory'
            ' in the parameter s3 bucket')
        yield ArgSpec(
            name='s3',
            type=str,
            default=None,
            help='s3 bucket to upload runs to; e.g., '
            'vlad-deeplearn. Use None to not upload.')
        yield ArgSpec(
            name='config',
            type=str,
            required=True,
            help='yaml filename of variants to be grid-searched over')
        yield ArgSpec(
            name='ncpus',
            type=int,
            default=0,
            help='cpus per task')
        yield ArgSpec(
            name='ngpus',
            type=int,
            default=1,
            help='gpus per task')

    def __init__(self):
        super().__init__('tune', 'ray tuning',
                         list(TuneFlags._generate_arguments()))


def _verify_s3_newdir(bucket_name, experiment_name):
    if bucket_name is None:
        return
    s3 = boto3.resource('s3')
    try:
        s3.meta.client.head_object(Bucket=bucket_name, Key=experiment_name)
        raise ValueError('key {} exists in bucket {} already'.format(
            experiment_name, bucket_name))
    except ClientError as exc:
        if exc.response['Error']['Code'] != '404':
            raise exc


def _verify_s3(bucket_name):
    if bucket_name is None:
        return
    s3 = boto3.resource('s3')
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 403:
            raise ValueError('Forbidden access to bucket {}'.format(
                bucket_name))
        elif error_code == 404:
            raise ValueError('Bucket {} does not exist'.format(bucket_name))
        else:
            raise e


def ray_train(config, status_reporter):
    """
    Entry point called by ray hypers tuning, remotely.

    config should be a dictionary with the usual flags for main_X.py
    X is specified by config['main'] (currently only
    X = 'ddpg' or 'sac' are supported).

    This then basically runs

    python main_X.py <config dictionary as flags>

    The other arguments are determined by their keys (flag) and values
    (argument for the flag). If the value is None then that flag gets no
    argument.

    Note that config should also have a ray_num_gpus argument (not related to
    flags, but related to launching with the appropriate gpus)
    """
    # Make sure we only ever use our assigned GPU:
    # per https://ray.readthedocs.io/en/latest/using-ray-with-gpus.html
    # Use the trick described here to shard GPUs:
    # https://github.com/ray-project/ray/issues/402
    num_gpus = max(ngpus(), 1)
    gpu_ids = [str(gpuid % num_gpus) for gpuid in ray.get_gpu_ids()]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

    if config['main'] == 'ddpg':
        flags = ALL_DDPG_FLAGS
        train_fn = ddpg_train
    elif config['main'] == 'sac':
        flags = ALL_SAC_FLAGS
        train_fn = sac_train
    else:
        raise ValueError('{} unrecognized main'.format(config['main']))

    del config['main']

    args = []
    for k, v in config.items():
        args.append('--' + k)
        if v is not None:
            if isinstance(v, str):
                args.append(v)
            else:
                args.append(repr(v))

    parsed_flags = parse_args(flags, args)

    def _report_hook(summaries, stats):
        if summaries is None:
            status_reporter(
                timesteps_total=reporter.timestep() + 1,
                done=1)
            return

        kwargs = {}
        if 'current policy reward' in stats:
            kwargs['episode_reward_mean'] = np.mean(stats[
                'current policy reward'])
            status_reporter(
                timesteps_total=reporter.timestep(),
                done=0,
                info=summaries,
                **kwargs)

    with reporter.report_hook(_report_hook):
        with setup_experiment_context(parsed_flags):
            train_fn()


def _main(args):
    _verify_s3(args.tune.s3)
    _verify_s3_newdir(args.tune.s3, args.tune.experiment_name)

    if args.tune.self_host:
        ray.init(num_gpus=max(ngpus(), 1))
    else:
        ip = ray.services.get_node_ip_address()
        ray.init(redis_address=(ip + ':' + args.tune.port))

    register_trainable("ray_train", ray_train)

    with open(args.tune.config) as f:
        config = yaml.load(f)

    experiment_setting = {
        'run': 'ray_train',
        'trial_resources': {'cpu': args.tune.ncpus, 'gpu': args.tune.ngpus},
        'stop': {'done': 1},
        'config': config,
        'local_dir': './data'
    }
    assert 'main' in config
    assert config['main'] in ['ddpg', 'sac'], config['main']
    if args.tune.s3 is not None:
        bucket_path = 's3://' + args.tune.s3 + '/'
        bucket_path += args.tune.experiment_name
        experiment_setting['upload_dir'] = bucket_path

    scheduler = None
    if args.tune.median_stop > 0:
        scheduler = MedianStoppingRule(
            time_attr='timesteps_total',
            grace_period=args.tune.median_stop,
            reward_attr='episode_reward_mean')
    run_experiments(
        {args.tune.experiment_name: experiment_setting},
        server_port=int(args.tune.server_port),
        with_server=True,
        scheduler=scheduler)


if __name__ == "__main__":
    _args = parse_args([TuneFlags()])
    _main(_args)
