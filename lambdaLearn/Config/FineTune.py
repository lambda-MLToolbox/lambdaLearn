from lambdaLearn.Network.Inc_Net import IncrementalNet

##### must only method-related, can't be dataset-related #####


model_name = "resnet32"  # backbone
seed = 1989
# dataset = "cifar100"

network = IncrementalNet(model_name, seed=seed)

memory_size = 0
memory_per_class = 0
fixed_memory = False

device = [0]  # set to -1 for CPU

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 80
lrate = 0.1
milestones = [40, 70]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
