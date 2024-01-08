from lambdaLearn.Network.Inc_Net import IncrementalNet

model_name = "resnet32"
seed = 1989

network = IncrementalNet(model_name, seed=seed)

memory_size = 0
memory_per_class = 0
fixed_memory = False

device = [0]

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 180
lrate = 0.1
milestones = [70, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2
lamda = 1000
fishermax = 0.0001