import threading
import subprocess

""" Main function """
import logging
from random import seed

import configargparse
from swarmI4.experiment import *

# Since some arguments are local to individual components (see for instance map generators), we ise the singleton
# argument parser from the configargparse lib.

parser = configargparse.get_arg_parser()  # ArgumentParser(description="Swarm robotics for I4.0 abstract simulator.")


def parse_args():
    """ Handles the arguments """
    # parser = configargparse.get_arg_parser()

    parser.add('-c', '--config', is_config_file=True, help='Config file')

    # parser.add_argument("-c", "--config",
    #                     help="Config file",
    #                     nargs=1, metavar="output_file",
    #                     default="")

    parser.add_argument("-o", "--output_path",
                        help="path to output",
                        nargs=1, metavar="output_path",
                        default="out/")

    return parser.parse_known_args()


def run_test(n_test, out_path, conf_path):
    subprocess.run(["python", "__main__.py", "-c", conf_path, "-o", (out_path + str(n_test) + "/")])


if __name__ == "__main__":
    arg, _ = parse_args()

    if type(arg.config) == list:
        arg.config = arg.config[0]

    if type(arg.output_path) == list:
        arg.output_path = arg.output_path[0]

    num_runs = 2

    threads = list()

    # main(arg)
    for i in range(num_runs):
        thread = threading.Thread(target=run_test, args=(i, arg.output_path, arg.config))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
