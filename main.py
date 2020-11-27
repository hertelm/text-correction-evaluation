import time
from typing import List, Tuple, Iterable, Set, Optional, Dict, Iterator

from enum import Enum
import numpy as np
from termcolor import colored
from functools import lru_cache
import argparse
import multiprocessing as mp

from helper.files import read_lines, read_file
from helper.data_structures import izip
from edit_distance.transposition_edit_distance import edit_operations as get_edit_operations, EditOperation, \
    OperationType
from evaluation.token_operations import assign_edits_to_tokens
from tokenization.regex_tokenizer import RegexTokenizer
from helper.pickle import load_object


@lru_cache(6)
def edit_operations(a: str, b: str, space_replace: bool) -> List[EditOperation]:
    return get_edit_operations(a, b, space_replace)


def match_tokens(a: List[str], b: List[str]) -> List[Tuple[int, int]]:
    n_a = len(a)
    n_b = len(b)
    d = np.zeros(shape=(n_a + 1, n_b + 1), dtype=int)
    # d[:, 0] = range(d.shape[0])
    # d[0, :] = range(d.shape[1])
    is_match = np.zeros_like(d, dtype=bool)
    for i in range(n_a):
        for j in range(n_b):
            d[i + 1, j + 1] = max(
                d[i, j + 1],
                d[i + 1, j],
                d[i, j]
            )
            if a[i] == b[j] and d[i + 1, j + 1] <= d[i, j] + 1:
                d[i + 1, j + 1] = d[i, j] + 1
                is_match[i + 1, j + 1] = True
    matchings = []
    i, j = n_a, n_b
    while i > 0 and j > 0:
        if is_match[i, j]:
            i, j = i - 1, j - 1
            matchings.append((i, j))
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j]:
            i = i - 1
        elif j > 0 and d[i, j] == d[i, j - 1]:
            j = j - 1
        else:
            raise Exception("something unexpected happened")
    return matchings[::-1]


class Label(Enum):
    EDIT = 0
    MERGE = 1
    SPLIT = 2


LABEL_ABBREVIATIONS = {
    Label.EDIT: "E",
    Label.MERGE: "M",
    Label.SPLIT: "S"
}


def is_split_operation(operation: EditOperation) -> bool:
    return operation.type == OperationType.INSERTION and operation.character == ' '


def is_merge_operation(operation: EditOperation) -> bool:
    return operation.type == OperationType.DELETION and operation.character == ' '


def operation2label(operation: EditOperation) -> Label:
    if is_split_operation(operation):
        return Label.SPLIT
    if is_merge_operation(operation):
        return Label.MERGE
    return Label.EDIT


def inverse_label(label: Label) -> Label:
    if label == Label.EDIT:
        return label
    return Label.SPLIT if label == Label.MERGE else Label.MERGE


def token_labels(a: str, b: str, inverse: bool) -> List[List[Label]]:
    edit_ops = edit_operations(a, b, space_replace=False)
    token_ops = assign_edits_to_tokens(edit_ops, a.split())
    labels = []
    for _ in token_ops:
        labels.append([])
    for t_i, ops in enumerate(token_ops):
        for op in ops:
            label = operation2label(op)
            labels[t_i].append(label if not inverse else inverse_label(label))
            if label == Label.MERGE:
                labels[t_i + 1].append(label if not inverse else inverse_label(label))
    return labels


def ground_truth_token_labels(correct: str, corrupt: str) -> List[List[Label]]:
    return token_labels(correct, corrupt, inverse=False)


def input_true_token_labels(correct: str, corrupt: str) -> List[List[Label]]:
    return token_labels(corrupt, correct, inverse=True)


def input_predicted_token_labels(corrupt: str, predicted: str) -> List[List[Label]]:
    return token_labels(corrupt, predicted, inverse=True)


def is_token_merged(token_ops: List[EditOperation]) -> bool:
    return len(token_ops) > 0 and is_merge_operation(token_ops[-1])


def n_split_operations(token_ops: List[EditOperation]):
    return sum(1 if is_split_operation(op) else 0 for op in token_ops)


def group_tokens(a: str, b: str) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    a_tokens = a.split()
    edit_ops = edit_operations(a, b, space_replace=False)
    token_ops = assign_edits_to_tokens(edit_ops, a_tokens)
    matchings = []
    b_i = 0
    a_group = []
    b_group = []
    for a_i, ops in enumerate(token_ops):
        a_group.append(a_i)
        n_splits = n_split_operations(ops)
        if is_token_merged(ops):
            b_group.extend(range(b_i, b_i + n_splits))
            b_i += n_splits
        else:
            b_group.extend(range(b_i, b_i + n_splits + 1))
            b_i += n_splits + 1
            matchings.append((tuple(a_group), tuple(b_group)))
            a_group = []
            b_group = []
    return matchings


def all_in_set(elements: Iterable, set: Set) -> bool:
    n = 0
    for element in elements:
        n += 1
        if element not in set:
            return False
    if n == 0:
        return False
    return True


class TokenLabel(Enum):
    NONE = 0
    SINGLE_EDIT = 1
    MULTI_EDIT = 2
    SPLIT = 3
    MERGE = 4
    MIXED = 5

    def __lt__(self, other):
        return self.value < other.value


class EvaluationCase(Enum):
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1
    FALSE_NEGATIVE = 2
    DID_DETECT = 3
    WAS_DETECTED = 4
    UNDETECTED = 5
    PREDICTED = 6

    def __lt__(self, other):
        return self.value < other.value


def edit_labels2token_label(labels: List[Label]) -> TokenLabel:
    if len(labels) == 0:
        return TokenLabel.NONE
    elif len(labels) == 1:
        if labels[0] == Label.EDIT:
            return TokenLabel.SINGLE_EDIT
        if labels[0] == Label.SPLIT:
            return TokenLabel.SPLIT
        if labels[0] == Label.MERGE:
            return TokenLabel.MERGE
    else:
        for label in labels:
            if label != Label.EDIT:
                return TokenLabel.MIXED
    return TokenLabel.MULTI_EDIT


def fraction(correct: int, total: int) -> float:
    if total == 0:
        return 0
    return correct / total


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def column(text: str, width: int) -> str:
    return "& " + text + ' ' * (width - len(text) - 1)


def percent_column(value: float, width: int) -> str:
    return column("%.1f \\%%" % (value * 100), width)


class ErrorType(Enum):
    NONWORD = 0
    REAL_WORD = 1

    def __lt__(self, other):
        return self.value < other.value


ERROR_TYPE_ABBREVIATIONS = {
    ErrorType.NONWORD: "nonword",
    ErrorType.REAL_WORD: "real-word"
}

LABEL2LATEX = {
    None: "all",
    TokenLabel.SINGLE_EDIT: "single-edit",
    TokenLabel.MULTI_EDIT: "multi-edit",
    TokenLabel.SPLIT: "split",
    TokenLabel.MERGE: "merge",
    TokenLabel.MIXED: "mixed",
    ErrorType.NONWORD: "nonword",
    ErrorType.REAL_WORD: "real-word"
}


def get_error_type(text: str, tokenizer: RegexTokenizer, words: Set[str]) -> ErrorType:
    tokens = tokenizer.words(text)
    is_editable = False
    if len(tokens) == 0:
        return ErrorType.NONWORD
    for t in tokens:
        if tokenizer.editable(t):
            is_editable = True
            if t not in words:
                return ErrorType.NONWORD
    return ErrorType.REAL_WORD if is_editable else ErrorType.NONWORD


def error_types(correct: str,
                corrupt: str,
                tokenizer: RegexTokenizer,
                words: Set[str]) -> Tuple[List[ErrorType],
                                          List[ErrorType],
                                          List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    corrupt_tokens = corrupt.split()
    matchings = group_tokens(correct, corrupt)
    correct_error_types = []
    corrupt_error_types = []
    for correct_indices, corrupt_indices in matchings:
        matched_corrupt_tokens = [corrupt_tokens[i] for i in corrupt_indices]
        matched_corrupt_error_types = [get_error_type(token, tokenizer, words) for token in matched_corrupt_tokens]
        correct_error_type = ErrorType.NONWORD if ErrorType.NONWORD in matched_corrupt_error_types else ErrorType.REAL_WORD
        corrupt_error_types.extend(matched_corrupt_error_types)
        correct_error_types.extend([correct_error_type] * len(correct_indices))
    return correct_error_types, corrupt_error_types, matchings


class Evaluator:
    def __init__(self):
        self.count = {
            label: {
                case: 0 for case in EvaluationCase
            } for label in list(TokenLabel) + list(ErrorType)
        }

    def add(self, labels: List[Label], case: EvaluationCase, error_type: ErrorType):
        label = edit_labels2token_label(labels)
        self.count[label][case] += 1
        self.count[error_type][case] += 1

    def _get_counts(self, label: Optional[TokenLabel] = None):
        total = {case: 0 for case in EvaluationCase}
        if label is None:
            labels = list(TokenLabel)
        else:
            labels = [label]
        for label in labels:
            for case in EvaluationCase:
                total[case] += self.count[label][case]
        return total

    def print_statistics(self):
        for label in [None] + sorted(TokenLabel):
            counts = self._get_counts(label)
            for case in sorted(EvaluationCase):
                print("ALL" if label is None else label.name,
                      case.name,
                      counts[case])

    def print_table(self, latex: bool):
        for label in [None] + sorted(ErrorType) + sorted(TokenLabel):
            if label == TokenLabel.NONE:
                continue
            counts = self._get_counts(label)
            correction_precision = fraction(counts[EvaluationCase.TRUE_POSITIVE],
                                            counts[EvaluationCase.TRUE_POSITIVE] + counts[
                                                EvaluationCase.FALSE_POSITIVE])
            correction_recall = fraction(counts[EvaluationCase.TRUE_POSITIVE],
                                         counts[EvaluationCase.TRUE_POSITIVE] + counts[EvaluationCase.FALSE_NEGATIVE])
            correction_f1 = f1(correction_precision, correction_recall)
            detection_precision = fraction(counts[EvaluationCase.DID_DETECT],
                                           counts[EvaluationCase.PREDICTED])
            detection_recall = fraction(counts[EvaluationCase.WAS_DETECTED],
                                        counts[EvaluationCase.WAS_DETECTED] + counts[EvaluationCase.UNDETECTED])
            detection_f1 = f1(detection_precision, detection_recall)
            label = LABEL2LATEX[label]
            column_width = 17
            if latex:
                print((' ' * 26 + "%s" * 7 + "\\\\") % (
                    column(label, 13),
                    percent_column(detection_precision, column_width),
                    percent_column(detection_recall, column_width),
                    percent_column(detection_f1, column_width),
                    percent_column(correction_precision, column_width),
                    percent_column(correction_recall, column_width),
                    percent_column(correction_f1, column_width)
                ))
            else:
                print("%s | %.2f (%i/%i) %.2f (%i/%i) %.2f | %.2f (%i/%i) %.2f (%i/%i) %.2f" % (
                    label,
                    detection_precision * 100,
                    counts[EvaluationCase.DID_DETECT],
                    counts[EvaluationCase.PREDICTED],
                    detection_recall * 100,
                    counts[EvaluationCase.WAS_DETECTED],
                    counts[EvaluationCase.WAS_DETECTED] + counts[EvaluationCase.UNDETECTED],
                    detection_f1 * 100,
                    correction_precision * 100,
                    counts[EvaluationCase.TRUE_POSITIVE],
                    counts[EvaluationCase.TRUE_POSITIVE] + counts[EvaluationCase.FALSE_POSITIVE],
                    correction_recall * 100,
                    counts[EvaluationCase.TRUE_POSITIVE],
                    counts[EvaluationCase.TRUE_POSITIVE] + counts[EvaluationCase.FALSE_NEGATIVE],
                    correction_f1 * 100
                ))

    def print_evaluation(self):
        self.print_table(latex=False)


def index2group_matching(groupings: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> Dict[int, Tuple[int]]:
    i2group = {}
    for left, right in groupings:
        for i in left:
            i2group[i] = right
    return i2group


def any_edited(group: Tuple[int, ...], labels: List[List[Label]]):
    for i in group:
        if len(labels[i]) > 0:
            return True
    return False


def evaluate_sample(correct: str, corrupt: str, predicted: str):
    evaluations = []

    is_correct = int(predicted == correct)

    correct_tokens = correct.split()
    corrupt_tokens = corrupt.split()
    predicted_tokens = predicted.split()

    n_gt = len(correct_tokens)
    n_in = len(corrupt_tokens)
    n_pred = len(predicted_tokens)

    # matchings
    pred_gt_matching = match_tokens(predicted_tokens, correct_tokens)
    in_gt_matching = match_tokens(corrupt_tokens, correct_tokens)
    pred_in_matching = match_tokens(predicted_tokens, corrupt_tokens)

    # gt labels
    gt_correctly_predicted = set(i for _, i in pred_gt_matching)
    # gt_not_corrupt = set(i for _, i in in_gt_matching)
    gt_labels = ground_truth_token_labels(correct, corrupt)
    gt_not_corrupt = set(i for i, labels in enumerate(gt_labels) if len(labels) == 0)
    # print([label.name for label in gt_labels])

    # in labels
    # in_not_corrupt = set(i for i, _ in in_gt_matching)
    in_true_labels = input_true_token_labels(correct, corrupt)
    in_not_corrupt = set(i for i, labels in enumerate(in_true_labels) if len(labels) == 0)
    in_predicted_labels = input_predicted_token_labels(corrupt, predicted)

    # pred labels
    pred_correct = set(i for i, _ in pred_gt_matching)
    pred_unchanged = set(i for i, _ in pred_in_matching)

    # error types
    gt_error_types, in_error_types, gt2in_grouping = error_types(correct, corrupt, tokenizer, words)
    gt2inputs = index2group_matching(gt2in_grouping)

    # GROUND TRUTH SEQUENCE

    contains_gt_none = False
    gt_sequence = ""
    for gt_i, gt_token, labels, error_type in izip(correct_tokens, gt_labels, gt_error_types):
        color = None
        if gt_i not in gt_not_corrupt:
            if gt_i in gt_correctly_predicted and any_edited(gt2inputs[gt_i], in_predicted_labels):
                color = "green"
                # evaluator.add(labels, EvaluationCase.TRUE_POSITIVE, error_type)
                # queue.put((labels, EvaluationCase.TRUE_POSITIVE, error_type))
                evaluations.append((labels, EvaluationCase.TRUE_POSITIVE, error_type))
            else:
                color = "yellow"
                # evaluator.add(labels, EvaluationCase.FALSE_NEGATIVE, error_type)
                evaluations.append((labels, EvaluationCase.FALSE_NEGATIVE, error_type))
                # queue.put((labels, EvaluationCase.FALSE_NEGATIVE, error_type))
        elif gt_i not in gt_correctly_predicted:
            color = "red"
        if gt_i > 0:
            gt_sequence += ' '
        text = gt_token
        if len(labels) > 0:
            text += "[%s]" % edit_labels2token_label(labels).name
            text += "{%s}" % ERROR_TYPE_ABBREVIATIONS[error_type]
        gt_sequence += colored(text, color)

    # INPUT SEQUENCE

    in_correctly_predicted = set()
    for corrupt_group, predicted_group in group_tokens(corrupt, predicted):
        if all_in_set(predicted_group, pred_correct):
            for in_i in corrupt_group:
                in_correctly_predicted.add(in_i)

    in_sequence = ""
    for in_i, in_token, true_labels, predicted_labels, error_type in \
            izip(corrupt_tokens, in_true_labels, in_predicted_labels, in_error_types):
        if in_i > 0:
            in_sequence += ' '
        text = in_token
        if len(predicted_labels) > 0 or len(true_labels) > 0:
            text += "[%s->%s]" % (edit_labels2token_label(true_labels).name,
                                  edit_labels2token_label(predicted_labels).name)
            text += "{%s}" % ERROR_TYPE_ABBREVIATIONS[error_type]
        color = None
        corrupt = len(true_labels) > 0
        changed = len(predicted_labels) > 0
        correct = in_i in in_correctly_predicted
        if corrupt:
            if changed:
                color = "green" if correct else "blue"
                # evaluator.add(predicted_labels, EvaluationCase.DID_DETECT, error_type)
                # evaluator.add(predicted_labels, EvaluationCase.PREDICTED, error_type)
                # evaluator.add(true_labels, EvaluationCase.WAS_DETECTED, error_type)
                # queue.put((predicted_labels, EvaluationCase.DID_DETECT, error_type))
                # queue.put((predicted_labels, EvaluationCase.PREDICTED, error_type))
                # queue.put((true_labels, EvaluationCase.WAS_DETECTED, error_type))
                evaluations.append((predicted_labels, EvaluationCase.DID_DETECT, error_type))
                evaluations.append((predicted_labels, EvaluationCase.PREDICTED, error_type))
                evaluations.append((true_labels, EvaluationCase.WAS_DETECTED, error_type))
            else:
                color = "yellow"
                # evaluator.add(true_labels, EvaluationCase.UNDETECTED, error_type)
                # queue.put((true_labels, EvaluationCase.UNDETECTED, error_type))
                evaluations.append((true_labels, EvaluationCase.UNDETECTED, error_type))
        else:
            if changed:
                color = "red"
                # evaluator.add(predicted_labels, EvaluationCase.PREDICTED, error_type)
                # queue.put((predicted_labels, EvaluationCase.PREDICTED, error_type))
                evaluations.append((predicted_labels, EvaluationCase.PREDICTED, error_type))
        if changed and not correct:
            # evaluator.add(predicted_labels, EvaluationCase.FALSE_POSITIVE, error_type)
            # queue.put((predicted_labels, EvaluationCase.FALSE_POSITIVE, error_type))
            evaluations.append((predicted_labels, EvaluationCase.FALSE_POSITIVE, error_type))
        in_sequence += colored(text, color)

    # PREDICTED SEQUENCE

    pred_sequence = ""
    for pred_i, pred_token in enumerate(predicted_tokens):
        color = None
        if pred_i not in pred_unchanged and pred_i in pred_correct:
            color = "green"
        elif pred_i in pred_unchanged and pred_i not in pred_correct:
            color = "yellow"
        elif pred_i not in pred_unchanged and pred_i not in pred_correct:
            color = "red"
        if pred_i > 0:
            pred_sequence += ' '
        pred_sequence += colored(pred_token, color)

    print("GROUND TRUTH:\n" + gt_sequence)
    print("INPUT:\n" + in_sequence)
    print("PREDICTED:\n" + pred_sequence)
    print()

    return evaluations, is_correct


def evaluate_sequences(evaluator: Evaluator,
                       correct_sequences: Iterator[str],
                       corrupt_sequences: Iterator[str],
                       predicted_sequences: Iterator[str],
                       n: int):
    n_correct = 0
    total_sequences = 0
    for s_i, correct, corrupt, predicted in izip(correct_sequences,
                                                 corrupt_sequences,
                                                 predicted_sequences):
        if s_i == n:
            break

        evaluations, is_correct = evaluate_sample(correct, corrupt, predicted)

        total_sequences += 1
        n_correct += is_correct

        for labels, case, error_type in evaluations:
            evaluator.add(labels, case, error_type)

    return n_correct, total_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the predictions of a spell checking program on a given"
                                                 " benchmark of misspelled and correct text.")
    parser.add_argument("--correct", required=True, type=str,
                        help="Path to the file with the ground truth sequences.")
    parser.add_argument("--misspelled", required=True, type=str,
                        help="Path to the file with the misspelled sequences.")
    parser.add_argument("--predictions", required=True, type=str,
                        help="Path to the file with the predicted sequences.")
    parser.add_argument("--words", type=str, default="data/words.txt",
                        help="Path to the vocabulary (one word per line, default: data/words.txt).")
    parser.add_argument("-n", type=int, default=-1,
                        help="Number of sequences to evaluate (default: all).")
    parser.add_argument("-mp", action="store_true")
    args = parser.parse_args()

    words = set(read_lines(args.words))
    print(len(words))

    tokenizer = RegexTokenizer()
    evaluator = Evaluator()

    start = time.monotonic()

    if args.mp:
        correct_sequences = read_file(args.correct)[:args.n]
        corrupt_sequences = read_file(args.misspelled)[:args.n]
        predicted_sequences = read_file(args.predictions)[:args.n]

        n_correct = 0
        total_sequences = 0

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(evaluate_sample,
                                   list(zip(correct_sequences, corrupt_sequences, predicted_sequences)))

        for evaluations, is_correct in results:
            for labels, case, error_type in evaluations:
                evaluator.add(labels, case, error_type)
            n_correct += is_correct
            total_sequences += 1

    else:
        correct_sequences = read_lines(args.correct)
        corrupt_sequences = read_lines(args.misspelled)
        predicted_sequences = read_lines(args.predictions)

        n_correct, total_sequences = evaluate_sequences(evaluator,
                                                        correct_sequences,
                                                        corrupt_sequences,
                                                        predicted_sequences,
                                                        args.n)
    evaluator.print_evaluation()
    end = time.monotonic()
    print(f"Processing {total_sequences} sequences took {end - start:.2f} seconds")
    print()
    print("%.1f sequence accuracy" % ((n_correct / total_sequences) * 100))
