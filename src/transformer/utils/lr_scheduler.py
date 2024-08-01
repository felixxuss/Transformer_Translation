import logging

logger = logging.getLogger(__name__)

class LRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4_000, step_size=1) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_size = step_size

        self.curr_step = 0
        self._lr = 0

        self._initialize_optimizer()

    def _calc_learning_rate(self, step=1) -> float:
        """Calculate the learning rate with the given step number

        Args:
            step (int, optional): Defaults to 1.

        Returns:
            float: Learning rate at the given step
        """
        return self.d_model**(-0.5) * min(step**(-0.5), step*self.warmup_steps**(-1.5))

    def step(self):
        """Update the learning rate
        """
        self.curr_step += self.step_size
        self._lr = self._calc_learning_rate(self.curr_step)
        logger.debug(f"LR at step {self.curr_step}: {self._lr:.4e}")

        # update lr in parameter groups
        # the learning rate is stored in optimizer.param_groups[i]['lr']
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr

    def _initialize_optimizer(self):
        """Initialize the learning rate to the initial value
        """
        self._lr = self._calc_learning_rate()
        logger.debug(f"LR inizialized to: {self._lr:.4e}")

        # update lr in parameter groups
        # the learning rate is stored in optimizer.param_groups[i]['lr']
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr

    def get_last_lr(self):
        return self._lr