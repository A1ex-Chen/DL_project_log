from dataclasses import dataclass
from typing import List

from .. import Task
from ... import db


class DB(db.DB):

    @dataclass
    class Checkpoint(db.DB.Checkpoint):

        task_name: Task.Name = Task.Name.CLASSIFICATION

        @dataclass
        class Metrics(db.DB.Checkpoint.Metrics):

            @dataclass
            class Overall(db.DB.Checkpoint.Metrics.Overall):
                accuracy: float
                avg_recall: float
                avg_precision: float
                avg_f1_score: float

            @dataclass
            class Specific(db.DB.Checkpoint.Metrics.Specific):
                categories: List[str]
                aucs: List[float]
                sensitivities: List[float]
                specificities: List[float]