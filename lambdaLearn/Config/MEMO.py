from lambdaLearn.Network.Inc_Net import AdaptiveNet


model_name = "memo_resnet32"
seed = 1989

network = AdaptiveNet(model_name, seed)

memory_size = 2000
memory_per_class = 20
fixed_memory = False
shuffle = True
init_cls = 10
increment = 10

device = [2, 3]

train_base = True
train_adaptive = False
debug = False
skip = False

csv_name='memo'

init_epoch = 200
init_lr = 0.1
init_weight_decay = 5e-4
init_lr_decay = 0.1
init_milestones = [60, 120, 170]

scheduler = 'steplr'
milestones = [80, 120, 150]
epochs = 170
lrate = 0.1
weight_decay = 2e-4
lrate_decay = 0.1

t_max = None
alpha_aux = 1.0


num_workers = 8
EPSILON = 1e-8
batch_size = 64