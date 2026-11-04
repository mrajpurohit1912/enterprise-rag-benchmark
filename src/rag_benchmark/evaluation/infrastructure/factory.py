

from typing import Any, Dict, List, Type

from rag_benchmark.evaluation.domain.base import EvaluationBase
from rag_benchmark.evaluation.infrastructure.generation import GenerationEvaluator
from rag_benchmark.evaluation.infrastructure.retrieval import RetrievalEvaluator
from rag_benchmark.evaluation.infrastructure.telemetry import TelemetryEvaluator


class EvaluatorFactory:
    """
    Enterprise Factory Pattern implementation for dynamically creating evaluation strategies.
    Stores classes (not instances) to allow lazy initialization and dynamic runtime configuration.
    """

    _registry: Dict[str, Type[EvaluationBase]] = {
        # Canonical short names
        "retrieval": RetrievalEvaluator,
        "generation": GenerationEvaluator,
        "telemetry": TelemetryEvaluator,
        # Aliases
        "retrieval_evaluator": RetrievalEvaluator,
        "generation_evaluator": GenerationEvaluator,
        "telemetry_evaluator": TelemetryEvaluator,
    }

    @classmethod
    def register(cls, eval_type: str, evaluator_cls: Type[EvaluationBase]) -> None:
        """Allows registering custom or plugin evaluation strategies at runtime."""
        cls._registry[eval_type.lower().strip()] = evaluator_cls

    @classmethod
    def create(cls, evaluator_name: str, **kwargs: Any) -> EvaluationBase:
        """
        Dynamically instantiates the requested evaluator with custom configuration parameters.

        Args:
            evaluator_name: Name of the evaluator (e.g. 'retrieval', 'generation', 'telemetry').
            **kwargs: Arguments passed directly to the concrete evaluator's __init__.

        Returns:
            An instance of EvaluationBase.

        Raises:
            ValueError: If evaluator_name is not registered.
        """
        key = evaluator_name.lower().strip()
        if key not in cls._registry:
            supported = ", ".join(sorted(set(cls._registry.keys())))
            raise ValueError(
                f"Unknown evaluator '{evaluator_name}'. Supported evaluators: [{supported}]"
            )

        evaluator_class = cls._registry[key]
        return evaluator_class(**kwargs)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Returns list of distinct supported evaluator types."""
        return sorted(list(set(cls._registry.keys())))

    