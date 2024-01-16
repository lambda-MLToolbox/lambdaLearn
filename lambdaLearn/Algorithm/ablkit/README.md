This folder contains an implementation for [**Abductive Learning (ABL)**](https://www.lamda.nju.edu.cn/publication/chap_ABL.pdf), which is built from [**ABLkit**](https://github.com/AbductiveLearning/ABLkit/tree/Dev), an efficient Python toolkit for ABL.

## What is ABL and ABLkit?

ABL is a novel paradigm that integrates machine learning and 
logical reasoning in a unified framework. It is suitable for tasks
where both data and (logical) domain knowledge are available. 

ABLkit is an efficient Python toolkit for ABL, which encapsulates advanced ABL techniques, providing users with an efficient and convenient toolkit to develop dual-driven ABL systems that leverage the power of both data and knowledge.

## (Optional) Install SWI-Prolog

If the use of a [Prolog-based knowledge base](https://ablkit.readthedocs.io/en/latest/Intro/Reasoning.html#prolog) is necessary, please install [SWI-Prolog](https://www.swi-prolog.org/):

For Linux users:

```bash
sudo apt-get install swi-prolog
```

For Windows and Mac users, please refer to the [SWI-Prolog Install Guide](https://github.com/yuce/pyswip/blob/master/INSTALL.md).

## How to Use

### Convert Data to ABLkit Form

ABLkit requires user data to be either structured as **tuple** `(X, gt_pseudo_label, Y)` or **ListData object** with `X`, `gt_pseudo_label` and `Y` attributes. 

`X` is a list of input examples containing instances, `gt_pseudo_label` is the ground-truth label of each example in `X` and `Y` is the ground-truth reasoning result of each example in `X`. Note that `gt_pseudo_label` is only used to evaluate the machine learning model's performance but not to train it. 

For tabular data, one may use `lambdaLearn.Algorithm.ablkit.data.DataConverter` 
to convert them to the aforementioned ABLkit data. Below shows an example:

```python
from lambdaLearn.Dataset.Tabular.BreastCancer import BreastCancer
from lambdaLearn.Algorithm.ablkit.data import DataConverter

# Get a lambdaLearn tabular dataset: BreastCancer
breast_dataset = BreastCancer(labeled_size=0.1, stratified=True, shuffle=True)

# Instantiate an DataConverter
dataconverter = DataConverter()

# Convert BreastCancer
ablkitdata = dataconverter.convert_lambdalearn_to_tuple(breast_dataset)
```

For other types of data, please manually convert them.

### Convert Model to ABLkit Form

ABLkit requires user to provide a base model for machine learning, and then wrap it into an instance of `ABLModel`. To convert LambdaLearn model to the aforementioned ABLkit learning models, one may use `lambdaLearn.Algorithm.ablkit.learning.ModelConverter`. Below shows an example:

```python
from torch import nn
from torch.optim import RMSprop, lr_scheduler
from lambdaLearn.Algorithm.SemiSupervised.Classification.FixMatch import FixMatch
from lambdaLearn.Network.LeNet5 import LeNet5
from lambdaLearn.Algorithm.ablkit.learning import ModelConverter

# Get a lambdaLearn model: FixMatch
model = FixMatch(
    network=LeNet5(),
    threshold=0.95,
    lambda_u=1.0,
    mu=7,
    T=0.5,
    epoch=1,
    num_it_epoch=2**20,
    num_it_total=2**20,
    device="cuda",
)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer_dict = dict(optimizer=RMSprop, lr=0.0003, alpha=0.9)
scheduler_dict = dict(scheduler=lr_scheduler.OneCycleLR, 
                      max_lr=0.0003, 
                      pct_start=0.15, 
                      total_steps=200)

# Instantiate a ModelConverter
converter = ModelConverter()

# Convert FixMatch
base_model = converter.convert_lambdalearn_to_basicnn(
    model, loss_fn=loss_fn, optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict
)
``` 

After completing the two conversions mentioned above, the construction of other components in a ABL system is the same as that of ABLkit. To explore detailed information of ABL and ABLkit, please refer to - [ABLkit's document](https://ablkit.readthedocs.io/en/latest/index.html).

## References

For more information about ABL, please refer to: [Zhou, 2019](http://scis.scichina.com/en/2019/076101.pdf) and [Zhou and Huang, 2022](https://www.lamda.nju.edu.cn/publication/chap_ABL.pdf).

```latex
@article{zhou2019abductive,
  title     = {Abductive learning: towards bridging machine learning and logical reasoning},
  author    = {Zhou, Zhi-Hua},
  journal   = {Science China Information Sciences},
  volume    = {62},
  number    = {7},
  pages     = {76101},
  year      = {2019}
}

@incollection{zhou2022abductive,
  title     = {Abductive Learning},
  author    = {Zhou, Zhi-Hua and Huang, Yu-Xuan},
  booktitle = {Neuro-Symbolic Artificial Intelligence: The State of the Art},
  editor    = {Pascal Hitzler and Md. Kamruzzaman Sarker},
  publisher = {{IOS} Press},
  pages     = {353--369},
  address   = {Amsterdam},
  year      = {2022}
}
```