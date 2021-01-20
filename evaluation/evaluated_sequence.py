from typing import Dict, List, Optional

import json

from helper.sortable_enum import SortableEnum


class TokenLabel(SortableEnum):
    NONE = 0
    SINGLE_EDIT = 1
    MULTI_EDIT = 2
    SPLIT = 3
    MERGE = 4
    MIXED = 5


class ErrorType(SortableEnum):
    NONWORD = 0
    REAL_WORD = 1


class EvaluationCase(SortableEnum):
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1
    FALSE_NEGATIVE = 2
    DID_DETECT = 3
    WAS_DETECTED = 4
    UNDETECTED = 5
    DAMAGED = 6


class EvaluatedToken:
    def __init__(self,
                 text: str,
                 error_type: Optional[ErrorType],
                 true_label: Optional[TokenLabel],
                 predicted_label: Optional[TokenLabel],
                 case: Optional[EvaluationCase]):
        self.text = text
        self.error_type = error_type
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.case = case

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "error_type": self.error_type.name if self.error_type is not None else None,
            "true_label": self.true_label.name if self.true_label is not None else None,
            "predicted_label": self.predicted_label.name if self.predicted_label is not None else None,
            "case": self.case.name if self.case is not None else None
        }


class EvaluatedSequence:
    def __init__(self,
                 correct_text: str,
                 misspelled_text: str,
                 predicted_text: str,
                 correct_tokens: List[EvaluatedToken],
                 misspelled_tokens: List[EvaluatedToken],
                 predicted_tokens: List[EvaluatedToken]):
        self.correct_text = correct_text
        self.misspelled_text = misspelled_text
        self.predicted_text = predicted_text
        self.correct_tokens = correct_tokens
        self.misspelled_tokens = misspelled_tokens
        self.predicted_tokens = predicted_tokens

    def to_dict(self):
        return {
            "correct": self.correct_text,
            "misspelled": self.misspelled_text,
            "predicted": self.predicted_text,
            "correct_tokens": [t.to_dict() for t in self.correct_tokens],
            "misspelled_tokens": [t.to_dict() for t in self.misspelled_tokens],
            "predicted_tokens": [t.to_dict() for t in self.predicted_tokens]
        }

    def to_json(self):
        return json.dumps(self.to_dict())
