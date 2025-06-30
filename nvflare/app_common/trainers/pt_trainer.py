from .trainer import Trainer

class PyTorchTrainer(Trainer):
    def __init__(self, script: str):
        super().__init__(script)

    def train(self):
        pass

    def evaluate(self):
        pass