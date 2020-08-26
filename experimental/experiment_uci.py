from src.experiment_tf import *
from src.utils import create_logger

date = "0826_new_preprossessing-2"
dataset_name = "housing"
kernel_order = 2
repetition = 3
selector = "horseshoe"

unique_name = create_unique_name(date,dataset_name, kernel_order, repetition, selector)

logger = create_logger("../log", unique_name, __file__)

run_train_and_test(date, dataset_name, selector=selector, kernel_order=kernel_order, repetition=repetition, plot_n_predict=False, logger=logger,
                   n_iter=10000,
                   lr=0.1)

# test from check point
# test_from_checkpoint(date, dataset_name, selector, kernel_order, repetition)

# dataset = load_data(dataset_name)
# print(0.6409190504562058*dataset.std_y_train)
# 5.48