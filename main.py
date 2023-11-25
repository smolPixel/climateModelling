from utils import Envirodataset
import yaml
import argparse


def main(argdict):
    train=Envirodataset(f'{argdict["path_data"]}/train.jsonl', argdict)
    dev=Envirodataset(f'{argdict["path_data"]}/dev.jsonl', argdict)
    test=Envirodataset(f'{argdict["path_data"]}/test.jsonl', argdict)

    print(len(train), len(dev), len(test))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    args = args.__dict__

    stream = open(args['config_file'], "r")
    argdict = yaml.safe_load(stream)

    print(argdict)
    main(argdict)
