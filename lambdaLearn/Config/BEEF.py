from lambdaLearn.Network.Inc_Net import BEEFISONet


model_name = "resnet32"  # backbone
seed = 1989

network = BEEFISONet(model_name, seed=seed)

memory_size = 2000
memory_per_class = 20
fixed_memory = True
shuffle = True
init_cls = 10
increment = 10

device = [0]

init_epochs = 10
init_lr = 0.1
init_weight_decay = 5e-4

lr = 0.1
batch_size = 128
weight_decay = 5e-4
num_workers = 8

expansion_epochs = 10
fusion_epochs = 5

logits_alignment = 1.7
energy_weight = 0.01
is_compress = False
reduce_batch_size = False

EPSILON = 1e-8
T = 2