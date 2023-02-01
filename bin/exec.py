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
from hydra import compose
from hydra import initialize as heinitialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None
from pathlib import Path


try:
    import sambaflow.samba as samba
    import sambaflow.samba.utils as sn_utils
    from sambaflow.mac.metadata import ConvTilingMetadata
    from sambaflow.samba import SambaTensor
    from sambaflow.samba.env import (disable_sgd_stoc,
                                enable_addbias_grad_accum_stoc,
                                enable_conv_grad_accum_stoc)
    from sambaflow.samba.test import consistency_test
    from sambaflow.samba.utils.argparser import parse_app_args
    #from sambaflow.samba.utils.common import common_app_driver
except:
    pass

import argparse
from typing import List, Tuple
#from cosmictagger.larcvio.larcv_fetcher import larcv_fetcher
#from larcvio.larcv_fetcher import larcv_fetcher
from src.utils.core.larcvio.larcv_fetcher import larcv_fetcher

# TODOBRW Begin This might get removed.
from src.networks.torch.uresnet2D import UResNet
from src.networks.torch.uresnet3D import UResNet3D

import torch.nn as nn
# TODOBRW Begin This might get removed.

#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

from src.config import Config, ComputeMode
from src.config.mode import ModeKind


def add_args(parser):
    parser.add_argument("overrides", nargs="*", default=[])

    # Network Params
    parser.add_argument("--model-type", type=str, default="uresnet2d", choices=["uresnet2d", "uresnet3d"])
    parser.add_argument("--use-bias", action="store_true", default=False)
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
    parser.add_argument("--downsampling", type=str, default="max_pooling", choices=["max_pooling", "convolutional"])
    parser.add_argument("--upsampling", type=str, default="convolutional")
    parser.add_argument("--connections", type=str, default="sum", choices=["sum", "concat"])
    parser.add_argument("--activation", type=str, default="leakyrelu")
    parser.add_argument("--network-depth", type=int, default=6)
    parser.add_argument('-b', "--batch-size", type=int, default=4)

    # Tiling Params
    parser.add_argument('--enable-conv-tiling', action="store_true", help='Enable DRAM tiling')
    parser.add_argument('--enable-stoc-rounding', action="store_true", help='Enable STOC Rounding')

    # Training Params
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003, help='Initial learning rate')
    parser.add_argument('--lr-schedule', action="store_true", help='Enable LR Scheduling')
    parser.add_argument('--loss-balance-scheme', type=str, choices=['none', 'focal', 'even', 'light'], default='focal')
    parser.add_argument('-i', '--iterations', type=int, default=100, help="Number of iterations to process")
    parser.add_argument('-m', '--compute-mode', type=str, choices=['CPU', 'RDU'], default='RDU', help="CPU or RDU")

    # DataLoader args
    parser.add_argument('--np-filepath', type=pathlib.Path, help="NP file output")
    parser.add_argument('-f', '--file', type=pathlib.Path, default="cosmic_tagging_train.h5", help="IO Input File")
    parser.add_argument('--synthetic', type=bool, default=False, help="Use synthetic data instead of real data.")
    parser.add_argument('-ds', '--downsample-images', default=1, type=int, help='Dense downsampling of the images.')
    parser.add_argument('--data-parallel', action="store_true", help='Distributed mode')

    # Helper args for accuracy regression
    parser.add_argument("--acc-test", action="store_true", default=False, help="Run the accuracy check")

    # Benchmarking and Profiling
    parser.add_argument("--run-benchmark",
        action="store_true",
                        default=False,
                        help="Profile and benchmark using samba profiler")
    parser.add_argument("--dataset-type",
                        type=str,
                        default="h5",
                        choices=["h5", "pickle", "ndarray"],
                        help="Datatype for dataset.")


def add_run_args(parser):

    # Tiling Params
    parser.add_argument('--enable-conv-tiling', action="store_true", help='Enable DRAM tiling')
    parser.add_argument('--enable-stoc-rounding', action="store_true", help='Enable STOC Rounding')

    # Training Params
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003, help='Initial learning rate')
    parser.add_argument('--lr-schedule', action="store_true", help='Enable LR Scheduling')
    parser.add_argument('--loss-balance-scheme', type=str, choices=['none', 'focal', 'even', 'light'], default='focal')
    parser.add_argument('-i', '--iterations', type=int, default=100, help="Number of iterations to process")
    parser.add_argument('-m', '--compute-mode', type=str, choices=['CPU', 'RDU'], default='RDU', help="CPU or RDU")


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
    """
        Dummy inputs for pef compile. This needs to be consistent with
        actual input dimensions.

    """
    image_shape = get_image_shape(args)
    input = samba.randn(*image_shape, name='input', batch_dim=0).bfloat16()
    input.requires_grad = not args.inference
    return (input, )


def get_image_shape(args: argparse.Namespace) -> List[int]:
    """
        Compute the image shape given the arguments.
        Based on the downsample image arguments the image shape changes.

    """
    full_height = larcv_fetcher.FULL_RESOLUTION_H
    full_width = larcv_fetcher.FULL_RESOLUTION_W
    channels = 3
    if args.model_type == "uresnet3d":
        return [args.batch_size, 1, channels
                ] + [int(i / (args.downsample_images + 1)) for i in [full_height, full_width]]
    else:
        return [args.batch_size, channels] + [int(i / (args.downsample_images + 1)) for i in [full_height, full_width]]


@consistency_test()
def test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
         outputs: List[samba.SambaTensor]) -> List[Tuple[str, SambaTensor]]:
    model.float()
    model.zero_grad()
    hyperparam_dict = {"lr": 0.0}
    inputs[0].grad = None
    gold_input = inputs[0].float()

    with samba.patched_torch():
        gold_outputs = model(gold_input)
    samba_outputs = samba.session.run(input_tensors=inputs, output_tensors=outputs, section_types=["fwd"])[0].float()
    output_vars = [("samba_outputs", samba_outputs), ("gold_outputs", gold_outputs)]

    print(f'gold_outputs_abs_sum: {gold_outputs.abs().sum().item()}')
    print(f'samba_outputs_abs_sum: {samba_outputs.abs().sum().item()}')

    sn_utils.assert_close(gold_outputs, samba_outputs, 'output', threshold=0.51, visualize=args.visualize)

    if not args.inference:
        output_grad = samba.randn_like(gold_outputs).bfloat16()
        gold_outputs.backward(output_grad.float())
        samba.session.run(input_tensors=inputs,
                          output_tensors=outputs,
                          grad_of_outputs=[output_grad],
                          section_types=["bckwd", "opt"],
                          hyperparam_dict=hyperparam_dict)[0]

        model.bfloat16()
        gold_input_grad = inputs[0].grad.bfloat16().float()
        samba_input_grad = inputs[0].sn_grad.float()
        print(f'gold_input_grad_abs_sum: {gold_input_grad.abs().sum().item()}')
        print(f'samba_input_grad_abs_sum: {samba_input_grad.data.abs().sum().item()}')
        sn_utils.assert_close(samba_input_grad, gold_input_grad, 'input_grad', threshold=1.4, visualize=args.visualize)

        gold_weight_grad = model.initial_convolution.conv.weight.grad.float()
        samba_weight_grad = model.initial_convolution.conv.weight.sn_grad.float()
        gold_bias_grad = model.initial_convolution.conv.bias.grad.float()
        samba_bias_grad = model.initial_convolution.conv.bias.sn_grad.float()

        print(f'gold_weight_grad_abs_sum: {gold_weight_grad.abs().sum()}')
        print(f'samba_weight_grad_abs_sum: {samba_weight_grad.abs().sum()}')
        print(f'gold_bias_grad_abs_sum: {gold_bias_grad.abs().sum()}')
        print(f'samba_bias_grad_abs_sum: {samba_bias_grad.abs().sum()}')

        # TODO(tejasn) : Add assertion checks for both weights and bias after 9-tile is fixed.

        output_vars += [("samba_input_grad", samba_input_grad), ("gold_input_grad", gold_input_grad)]
        output_vars += [(name + "_sn_grad", param.sn_grad) for name, param in model.named_parameters()]

    return output_vars


def app_setup(argparseArgs, hydraArgs):
    """
        Sets up app before mode based flows. This includes:
        1. samba-specific setup (stochartic rounding, conv tiling)
        2. app specific configs needed for compile (model init, optimizer init)
        3. inference mode handling is placed in here for perf -- this way
           we don't need to go through the setup overhead before inference
    """
    # Stochastic Rounding
    if argparseArgs.enable_stoc_rounding:
        enable_conv_grad_accum_stoc()
        enable_addbias_grad_accum_stoc()
    else:
        disable_sgd_stoc()

    assert hydraArgs.framework.name == "torch"
    # Begin from trainercore.__init__()
    if hydraArgs.framework.name == "torch":
        sparse = hydraArgs.framework.sparse
        io_dataformat = "channels_first"
    else:
        sparse = False

    _larcv_fetcher = larcv_fetcher(
        mode        = hydraArgs.mode.name.name,
        distributed = hydraArgs.run.distributed,
        downsample  = hydraArgs.data.downsample,
        # dataformat  = hydraArgs.data.data_format.name,
        dataformat  = io_dataformat,
        synthetic   = hydraArgs.data.synthetic,
        sparse      = sparse )
    # End from trainercore.__init__()




    # Instantiate model
    if argparseArgs.model_type == "uresnet3d":
        # model = UResNet3D(args)
        model = UResNet3D(hydraArgs.network, _larcv_fetcher.image_size())
    else:
        model = UResNet(hydraArgs.network)

    model.bfloat16()
    samba.from_torch_model_(model)

    # Dummy inputs required for tracing.
    inputs = get_inputs(argparseArgs)

    # Inference handling
    if argparseArgs.inference:
        model.eval()

    # Conv Tiling
    metadata = dict()
    if argparseArgs.enable_conv_tiling:
        original_size = inputs[0].shape
        metadata[ConvTilingMetadata.key] = ConvTilingMetadata(original_size=original_size)

    # Instantiate optimizer
    optim = samba.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9,
                                                                 0.997), weight_decay=0) if not argparseArgs.inference else None

    return model, inputs, optim, metadata


class exec(object):

    def __init__(self, config_file):

        parsed = Path(config_file)
        heinitialize(config_path=str(parsed.parent))
        overrides = ["mode=compile", "run.id=run_id", "run.distributed=False", "data.data_directory=/nvmedata/ANL/cosmictagger/", "data.downsample=0", "framework=torch", "run.compute_mode=CPU", "run.minibatch_size=1", "run.iterations=1", "run.precision=3"]
        config = compose(parsed.name, overrides=overrides)

        self.hydraArgs = config

        if config.mode.name == ModeKind.train:
            self.train()
        if config.mode.name == ModeKind.iotest:
            self.iotest()
        if config.mode.name == ModeKind.inference:
            self.inference()
        if config.mode.name == ModeKind.compile:
            self.compile()

    def init_mpi(self):
        if not self.hydraArgs.run.distributed:
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
            log_file = self.hydraArgs.output_dir + "/process.log"
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


    def compile(self):

        if self.hydraArgs.run.compute_mode == ComputeMode.RDU:
            sn_utils.set_seed(0)
            # Arg Handler -- note: no validity checking done here
            argv = sys.argv[1:]
            self.argparseArgs = parse_app_args(argv=argv, common_parser_fn=add_args)
            print(f'self.argparseArgs:\n{self.argparseArgs}')

        else:
            self.argparseArgs = None

        rank = self.init_mpi()

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.hydraArgs.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger()
        logger.info("Dumping launch arguments.")
        logger.info(sys.argv)

        if self.hydraArgs.run.compute_mode == ComputeMode.RDU:

            # Instantiate samba profiler.
            # We gate this by run_benchmark to avoid profiler overhead if we are
            # not using the samba profiler. Note: profiler start_event and end_event
            # have minimal overhead and don't need to be gated.
            if self.argparseArgs.run_benchmark:
                samba.session.start_samba_profile()

            # App setup
            app_setup_event = samba.session.profiler.start_event('app_setup')
            model, inputs, optim, metadata = app_setup(self.argparseArgs, self.hydraArgs)
            samba.session.profiler.end_event(app_setup_event)

            if self.argparseArgs.command == "compile":
                compile_event = samba.session.profiler.start_event('compile')
                samba.session.compile(model,
                                    inputs,
                                    optim,
                                    name='uresnet',
                                    app_dir=sn_utils.get_file_dir(__file__),
                                    metadata=metadata,
                                    init_output_grads=not self.argparseArgs.inference,
                                    config_dict=vars(self.argparseArgs),
                                    squeeze_bs_dim=True)
                samba.session.profiler.end_event(compile_event)



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

        if self.hydraArgs.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.1)
        for i in range(self.hydraArgs.run.iterations):
            start = time.time()
            mb = self.trainer.larcv_fetcher.fetch_next_batch("train", force_pop=True)

            end = time.time()

            logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")

        total_time = time.time() - global_start
        images_read = self.hydraArgs.run.iterations * self.hydraArgs.run.minibatch_size
        logger.info(f"Total IO Time: {total_time:.2f}s")
        logger.info(f"Total images read per batch: {self.hydraArgs.run.minibatch_size}")
        logger.info(f"Average Image IO Throughput: { images_read / total_time:.3f}")

    def make_trainer(self):


        if 'environment_variables' in self.hydraArgs.framework:
            for env in self.hydraArgs.framework.environment_variables.keys():
                os.environ[env] = self.hydraArgs.framework.environment_variables[env]

        if self.hydraArgs.mode.name == ModeKind.iotest:
            from src.utils.core import trainercore
            self.trainer = trainercore.trainercore(self.hydraArgs)
            return

        if self.hydraArgs.framework.name == "tensorflow":

            import logging
            logging.getLogger('tensorflow').setLevel(logging.FATAL)


            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # Import tensorflow and see what the version is.
            import tensorflow as tf

            if tf.__version__.startswith("2"):
                if self.hydraArgs.run.distributed:
                    from src.utils.tensorflow2 import distributed_trainer
                    self.trainer = distributed_trainer.distributed_trainer(self.hydraArgs)
                else:
                    from src.utils.tensorflow2 import trainer
                    self.trainer = trainer.tf_trainer(self.hydraArgs)
            else:
                if self.hydraArgs.run.distributed:
                    from src.utils.tensorflow1 import distributed_trainer
                    self.trainer = distributed_trainer.distributed_trainer(self.hydraArgs)
                else:
                    from src.utils.tensorflow1 import trainer
                    self.trainer = trainer.tf_trainer(self.hydraArgs)

        elif self.hydraArgs.framework.name == "torch":
            if self.hydraArgs.run.distributed:
                from src.utils.torch import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.hydraArgs)
            else:
                from src.utils.torch import trainer
                self.trainer = trainer.torch_trainer(self.hydraArgs)


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
        substring = s +  self.dictionary_to_str(self.hydraArgs)

        return substring




    def validate_arguments(self):

        from src.config.data import DataFormatKind

        logger = logging.getLogger()

        if self.hydraArgs.framework.name == "torch":
            # In torch, only option is channels first:
            if self.hydraArgs.data.data_format == DataFormatKind.channels_last:
                if self.hydraArgs.run.compute_mode == ComputeMode.GPU:
                    logger.warning("CUDA Torch requires channels_first, switching automatically")
                    self.hydraArgs.data.data_format = DataFormatKind.channels_first

        elif self.hydraArgs.framework.name == "tensorflow":
            if self.hydraArgs.mode.name == ModeKind.train:
                if self.hydraArgs.mode.quantization_aware:
                    logger.error("Quantization aware training not implemented in tensorflow.")

        self.hydraArgs.network.data_format = self.hydraArgs.data.data_format.name

# def getAllArgs(config_file: str):

#     parser = argparse.ArgumentParser()
#     add_args(parser)
#     args = parser.parse_args()

#     parsed = Path(config_file)
#     initialize(config_dir=str(parsed.parent), strict=False)
#     cfg = compose(parsed.name, overrides=args.overrides)

#     return cfg, args



#atsignhydra.main(config_path="../src/config", config_name="config")
#def main(cfg : OmegaConf) -> None:
def main(config_file: str) -> None:

    #cfg, args = getAllArgs(config_file)

    s = exec(config_file)


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra/job_logging=disabled']
    main("../src/config/config.yaml")
