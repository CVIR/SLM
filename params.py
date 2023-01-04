#All the required parameters for the framework.
src_gpu_id = 0
tgt_gpu_id = 0
pseudo_threshold = 0.3
dataset_name = "Office31"
batch_size = 64 

src_dataset = "amazon"
tgt_dataset = "webcam"

model_root = "./checkpoints"

log_step_freq = 100
eval_step_freq = 1
save_step_freq = 50
manual_seed = 1234

learning_rate = 0.0005
learning_rate_ws = 0.001
warmstart_models = 'True'
num_iter_warmstart = 5000
num_iter_adapt = 10000
momentum = 0.9

model_name = 'resnet50'
classifier_name = 'resnet50'