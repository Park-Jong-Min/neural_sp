"""Decode the ASR model."""

import argparse
import copy
import cProfile
import logging
import os
from setproctitle import setproctitle
import shutil
import sys
import time
import torch
from tqdm import tqdm

from neural_sp.bin.args_asr import parse_args_train
from neural_sp.bin.model_name import set_asr_model_name
from neural_sp.bin.train_utils import (
    compute_susampling_factor,
    load_checkpoint,
    load_config,
    save_config,
    set_logger,
    set_save_path
)
from neural_sp.datasets.asr import Dataset
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.data_parallel import CPUWrapperASR
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.trainers.lr_scheduler import LRScheduler
from neural_sp.trainers.optimizer import set_optimizer
from neural_sp.trainers.reporter import Reporter
from neural_sp.utils import mkdir_join
from neural_sp.bin.asr import evaluate

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

logger = logging.getLogger(__name__)

def main():

    args = parse_args_train(sys.argv[1:])
    args_init = copy.deepcopy(args)
    args_teacher = copy.deepcopy(args)
