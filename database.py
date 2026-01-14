import dataclasses
from strategies import ActiveLearningStrategy, ColdStartStrategy

class DataDatabase:
    def __init__(self):
        self.datasets: Datasets 
        self.experiments: Experiments
        self.llm_annotated_examples: LLMAnnotatedExamples
        ...
    
    def loadExperiment()
        
@dataclasses.dataclass
class Experiment:
    seed: int
    dataset: Dataset
    split: int 
    cold_start_strategy: ColdStartStrategy
    active_learning_strategy: ActiveLearningStrategy
    macro_f1: float
    accuracy: float

    @staticmethod
    def load_experiment(format: ExperimentFormat, experiment_config: ExperimentConfig) -> Experiment:
        ...

    @property
    def budget(self) -> int:
        return self.cold_start_strategy.budget + self.active_learning_strategy.budget
