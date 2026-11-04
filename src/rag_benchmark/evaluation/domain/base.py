from typing import List,Any
from abc import ABC,abstractmethod

from rag_benchmark.evaluation.domain.schemas import EvaluationSample

class EvaluationBase(ABC):
    @abstractmethod
    def evaluate(self,
        predictions:List[Any],
        ground_truth:List[EvaluationSample],
        )->Any:

        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Returns the list of metric names computed by this evaluator."""
        pass
    

    