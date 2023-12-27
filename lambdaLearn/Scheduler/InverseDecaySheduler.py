from lambdaLearn.Base.LambdaLR import LambdaLR


class InverseDecaySheduler(LambdaLR):
    def __init__(self, gamma=10, power=0.75, num_training_steps=1000):
        self.gamma=gamma
        self.power=power
        self.num_training_steps=num_training_steps
        super().__init__(lr_lambda=self._lr_lambda)

    def _lr_lambda(self, current_step):
        return (1 + self.gamma * min(1.0, current_step / float(self.num_training_steps))) ** (- self.power)
