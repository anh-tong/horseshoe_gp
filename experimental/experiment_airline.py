from experimental.experiment_tf import *

date = "0825"

dataset_name = "airline"
selector = "horseshoe"
kernel_order = 2
repetition = 1

run(date, dataset_name, selector=selector, kernel_order=kernel_order, repetition=repetition)

# test from check point
test_from_checkpoint(date, dataset_name, selector, kernel_order, repetition)