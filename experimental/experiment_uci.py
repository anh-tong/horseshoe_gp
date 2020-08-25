from experimental.experiment_tf import *

date = "0825"
dataset_name = "housing"
kernel_order = 2
repetition = 3
selector = "horseshoe"


run(date, dataset_name, selector=selector, kernel_order=kernel_order, repetition=repetition, plot_n_predict=False)

# test from check point
# test_from_checkpoint(date, dataset_name, selector, kernel_order, repetition)