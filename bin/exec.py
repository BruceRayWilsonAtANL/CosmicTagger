#!/usr/bin/env python
import os,sys,signal
import time
import pathlib
import logging
from logging import handlers

import numpy

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None


try:
    from sambaflow import samba

    import sambaflow.samba.utils as sn_utils
    from sambaflow.mac.metadata import ConvTilingMetadata
    from sambaflow.samba.utils.argparser import parse_app_args
    from sambaflow.samba.utils.common import common_app_driver
except:
    pass

import argparse
from typing import List, Tuple
#from cosmictagger.larcvio.larcv_fetcher import larcv_fetcher
from larcvio.larcv_fetcher import larcv_fetcher

# TODOBRW Begin This might get removed.
from src.networks.torch.uresnet2D import UResNet


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
    image_shape = get_image_shape(args)
    input = samba.randn(*image_shape, name='input', batch_dim=0).bfloat16()
    input.requires_grad = not args.inference
    return (input, )


def get_image_shape(args: argparse.Namespace) -> List[int]:
    """
    Compute the image shape given the arguments. Based on the downsample image arguments the image shape changes.
    """
    full_height = larcv_fetcher.FULL_RESOLUTION_H
    full_width = larcv_fetcher.FULL_RESOLUTION_W
    channels = 3
    return [args.batch_size, channels] + [int(i / (args.downsample_images + 1)) for i in [full_height, full_width]]
# TODOBRW Begin This might get removed.

#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

from src.config import Config, ComputeMode
from src.config.mode import ModeKind


def add_args(parser):
    # Network Params
    parser.add_argument("--use-bias", action="store_true", default=True)
    parser.add_argument("--batch-norm", action="store_true", default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--bottleneck-deepest", type=int, default=256)
    parser.add_argument("--block-concat", action="store_true", default=False)
    parser.add_argument("--filter-size-deepest", type=int, default=5)
    parser.add_argument("--blocks-deepest-layer", type=int, default=5)
    parser.add_argument("--blocks-per-layer", type=int, default=2)
    parser.add_argument("--blocks-final", type=int, default=0)
    parser.add_argument("--growth-rate", type=str, default="additive")
    parser.add_argument("--n-initial-filters", type=int, default=16)
    parser.add_argument("--downsampling", type=str, default="max_pooling")
    parser.add_argument("--upsampling", type=str, default="convolutional")
    parser.add_argument("--connections", type=str, default="sum")
    parser.add_argument("--network-depth", type=int, default=6)

    # DataLoader args
    parser.add_argument('-f', '--file', type=pathlib.Path, default="cosmic_tagging_train.h5", help="IO Input File")
    parser.add_argument('--synthetic', type=bool, default=False, help="Use synthetic data instead of real data.")
    parser.add_argument('-ds', '--downsample-images', default=1, type=int, help='Dense downsampling of the images.')

    # Helper args for accuracy regression
    parser.add_argument("--acc-test", action="store_true", default=False, help="Run the accuracy check")
    parser.add_argument(
        "--use-pickle-dataset",
        action="store_true",
        help="Use a preprocessed dataset for accuracy regressions. This flag will override other dataset options")


def add_run_args(parser):
    # Tiling Params
    parser.add_argument('--enable-tiling', type=bool, default=False, help='Enable DRAM tiling')

    # Training Params
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003, help='Initial learning rate')
    parser.add_argument('--loss-balance-scheme', type=str, choices=['none', 'focal', 'even', 'light'], default='focal')
    parser.add_argument('-i', '--iterations', type=int, default=100, help="Number of iterations to process")
    parser.add_argument('-m', '--compute-mode', type=str, choices=['CPU', 'RDU'], default='RDU', help="CPU or RDU")


def get_image_shape(args: argparse.Namespace) -> List[int]:
    """
    Compute the image shape given the arguments. Based on the downsample image arguments the image shape changes.
    """
    full_height = larcv_fetcher.FULL_RESOLUTION_H
    full_width = larcv_fetcher.FULL_RESOLUTION_W
    channels = 3
    return [args.batch_size, channels] + [int(i / (args.downsample_images + 1)) for i in [full_height, full_width]]


class exec(object):

    def __init__(self, config):

        self.args = config

        rank = self.init_mpi()

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger()
        logger.info("Dumping launch arguments.")
        logger.info(sys.argv)

        if self.args.run.compute_mode == ComputeMode.RDU:
            sn_utils.set_seed(256)
            self.argparseArgs = parse_app_args(argv=sys.argv, common_parser_fn=add_args, run_parser_fn=add_run_args)
        else:
            self.argparseArgs = None








        if self.args.run.compute_mode == ComputeMode.RDU:
            model = UResNet(self.args)

            model.bfloat16()

            samba.from_torch_(model)

            # Dummy inputs required for tracing.
            inputs = get_inputs(self.args)

            if self.args.inference:
                model.eval()

            metadata = dict()
            if self.args.enable_tiling:
                original_size = inputs[0].shape

                metadata[ConvTilingMetadata.key] = ConvTilingMetadata(original_size=original_size)

            # Instantiate a optimizer.
            optim = samba.optim.AdamW(model.parameters(), lr=0, betas=(0.9,
                                                                    0.997), weight_decay=0) if not self.args.inference else None

            if self.args.command == "compile":
                samba.session.compile(model,
                                    inputs,
                                    optim,
                                    name='uresnet',
                                    app_dir=sn_utils.get_file_dir(__file__),
                                    metadata=metadata,
                                    init_output_grads=not self.args.inference,
                                    config_dict=vars(self.args),
                                    squeeze_bs_dim=True)

            return


# run.distributed=False
        if config.mode.name == ModeKind.train:
            self.train()
        if config.mode.name == ModeKind.iotest:
            self.iotest()
        if config.mode.name == ModeKind.inference:
            self.inference()



    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm.Get_rank()


    def configure_logger(self, rank):

        logger = logging.getLogger()

        # Create a handler for STDOUT, but only on the root rank.
        # If not distributed, we still get 0 passed in here.
        if rank == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
            logger.addHandler(handler)

            # Add a file handler too:
            log_file = self.args.output_dir + "/process.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
            logger.addHandler(file_handler)

            logger.setLevel(logging.INFO)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


    def train(self):

        logger = logging.getLogger("cosmictagger")

        logger.info("Running Training")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()


    def iotest(self):

        self.make_trainer()
        logger = logging.getLogger("cosmictagger")

        logger.info("Running IO Test")
        logger.info(self.__str__())


        self.trainer.initialize(io_only=True)

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.1)
        for i in range(self.args.run.iterations):
            start = time.time()
            mb = self.trainer.larcv_fetcher.fetch_next_batch("train", force_pop=True)

            end = time.time()

            logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")

        total_time = time.time() - global_start
        images_read = self.args.run.iterations * self.args.run.minibatch_size
        logger.info(f"Total IO Time: {total_time:.2f}s")
        logger.info(f"Total images read per batch: {self.args.run.minibatch_size}")
        logger.info(f"Average Image IO Throughput: { images_read / total_time:.3f}")

    def make_trainer(self):


        if 'environment_variables' in self.args.framework:
            for env in self.args.framework.environment_variables.keys():
                os.environ[env] = self.args.framework.environment_variables[env]

        if self.args.mode.name == ModeKind.iotest:
            from src.utils.core import trainercore
            self.trainer = trainercore.trainercore(self.args)
            return

        if self.args.framework.name == "tensorflow":

            import logging
            logging.getLogger('tensorflow').setLevel(logging.FATAL)


            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # Import tensorflow and see what the version is.
            import tensorflow as tf

            if tf.__version__.startswith("2"):
                if self.args.run.distributed:
                    from src.utils.tensorflow2 import distributed_trainer
                    self.trainer = distributed_trainer.distributed_trainer(self.args)
                else:
                    from src.utils.tensorflow2 import trainer
                    self.trainer = trainer.tf_trainer(self.args)
            else:
                if self.args.run.distributed:
                    from src.utils.tensorflow1 import distributed_trainer
                    self.trainer = distributed_trainer.distributed_trainer(self.args)
                else:
                    from src.utils.tensorflow1 import trainer
                    self.trainer = trainer.tf_trainer(self.args)

        elif self.args.framework.name == "torch":
            if self.args.run.distributed:
                from src.utils.torch import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.args)
            else:
                from src.utils.torch import trainer
                self.trainer = trainer.torch_trainer(self.args, self.argparseArgs)


    def inference(self):


        logger = logging.getLogger("cosmictagger")

        logger.info("Running Inference")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()

    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                if hasattr(in_dict[key], "name"): attr = in_dict[key].name
                else: attr = in_dict[key]
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = attr,
                )
                substr += s
        return substr

    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring




    def validate_arguments(self):

        from src.config.data import DataFormatKind

        logger = logging.getLogger()

        if self.args.framework.name == "torch":
            # In torch, only option is channels first:
            if self.args.data.data_format == DataFormatKind.channels_last:
                if self.args.run.compute_mode == ComputeMode.GPU:
                    logger.warning("CUDA Torch requires channels_first, switching automatically")
                    self.args.data.data_format = DataFormatKind.channels_first

        elif self.args.framework.name == "tensorflow":
            if self.args.mode.name == ModeKind.train:
                if self.args.mode.quantization_aware:
                    logger.error("Quantization aware training not implemented in tensorflow.")

        self.args.network.data_format = self.args.data.data_format.name




@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(cfg : OmegaConf) -> None:

    s = exec(cfg)


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra/job_logging=disabled']
    main()
