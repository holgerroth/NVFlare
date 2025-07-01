from nvflare.job_config.script_runner import ScriptRunner

from .trainer import Trainer


class PyTorchTrainer(Trainer):
    def __init__(self, script: str, args: str = ""):

        self.script_args = args
        self.runner = ScriptRunner(script, args)

    def get_executor(self):
        return self.runner
