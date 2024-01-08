from lambdaLearn.Network.Inc_Net import IncrementalNetCosineSimple

model_name = "resnet32"
seed = 1989

network = IncrementalNetCosineSimple(model_name, seed=seed)

memory_size = 0
memory_per_class = 0
fixed_memory = False

device = [0]

init_epoch = 200
init_lr = 0.01
init_milestones = [80, 120]
init_lr_decay = 0.1
init_weight_decay = 0.0005

min_lr = 0

num_workers = 8
batch_size = 128  # TODO: 256 ?