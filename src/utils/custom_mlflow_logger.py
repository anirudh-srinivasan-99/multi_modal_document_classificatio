from typing import Any, Dict, Optional, Mapping
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only


class CustomMLFlowLogger(MLFlowLogger):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Custom MLFlow Logger that forces the X-axis to be the Epoch number.
        Inherits all arguments from standard MLFlowLogger.
        """
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_metrics(
        self, 
        metrics: Mapping[str, float], 
        step: Optional[int] = None
    ) -> None:
        """
        Overrides the default logging behavior to prioritize the 'epoch' 
        coordinate for the MLflow X-axis.
        """
        # 1. Capture the epoch from the metrics dict if Lightning provided it
        # Lightning usually injects 'epoch' into the dict during epoch-end hooks.
        epoch: Optional[int] = int(metrics.get("epoch", step)) if metrics.get("epoch") is not None else step

        # 2. Filter out 'epoch' from the dictionary so it doesn't log as its own chart
        metrics_to_log: Dict[str, float] = {
            k: float(v) for k, v in metrics.items() if k != "epoch"
        }

        # 3. Call the parent log_metrics using our forced 'epoch' as the step
        if epoch is not None:
            super().log_metrics(metrics_to_log, step=epoch)
        else:
            super().log_metrics(metrics_to_log, step=step)