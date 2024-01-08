from lambdaLearn.Network.Inc_Net import FOSTERNet

model_name = "resnet32"  # backbone
seed = 1989

network = FOSTERNet(model_name, seed=seed)

memory_size = 2000
memory_per_class = 20
fixed_memory = True
shuffle = True
init_cls = 50
increment = 10

device = [0]

beta1 = 0.96
beta2 = 0.97
oofc = "ft"
is_teacher_wa = False
is_student_wa = False
lambda_okd = 1
wa_value = 1
init_epochs = 200
init_lr = 0.1
init_weight_decay = 5e-4

lr = 0.1
batch_size = 128
weight_decay = 5e-4

boosting_epochs = 170
compression_epochs = 130

num_workers = 8
T = 2

EPSILON = 1e-8