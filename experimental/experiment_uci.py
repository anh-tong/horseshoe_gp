import os
import sys

from src.experiment_tf import *
from src.utils import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    date = sys.argv[1]
    dataset_name = sys.argv[2]
    kernel_order = 2
    repetition = 2
    selector = "horseshoe"

    unique_name = create_unique_name(date, dataset_name, kernel_order, repetition, selector)

    logger = create_logger("../log", unique_name, __file__)

    run_train_and_test(date, dataset_name, selector=selector, kernel_order=kernel_order, repetition=repetition,
                       logger=logger)

# test from check point
# test_from_checkpoint(date, dataset_name, selector, kernel_order, repetition)
