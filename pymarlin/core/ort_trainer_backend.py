from .trainer_backend import *
import sys

class ORTTrainerBackend(AbstractTrainerBackendDecorator):   
    def __init__(self, trainer_backend):
        super().__init__(trainer_backend)
    
    # TODO: add these under TrainerBackendDecoratorPassThrough, which ORT, Opacus can inherit from
    # so that DDP backend can get/set from wrapped SingleProcess*
    def __getattribute__(self, name):
        if name == 'trainer_backend':
           return super().__getattribute__(name)
        else:
            return self.trainer_backend.__getattribute__(name)

    def __setattr__(self, name, value):
        if name == 'trainer_backend':
            super().__setattr__(name, value)
        else:
            self.trainer_backend.__setattr__(name, value)
    
    def init(self, args: TrainerBackendArguments):
        try:
            from torch_ort import ORTModule
        except:
            self.logger.error("could not import ORTModule")
            sys.exit(1)
        
        # assert module interface has model defined (make standard?)
        assert(hasattr(self.trainer_backend.model, 'model'), 'self.trainer_backend.model.model does not exist')

        # assert module_interface.model is a torch.nn.module
        assert(isinstance(self.trainer_backend.model.model, torch.nn.module), "expected self.model property of type torch.nn.module")

        self.logger.info("Wrapping trainer_backend.model.model")
        self.trainer_backend.model.get_core_model = lambda: self.trainer_backend.model.model
        self.trainer_backend.model.model = ORTModule(self.trainer_backend.model.model)