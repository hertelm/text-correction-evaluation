from typing import List, Tuple, Iterable, Set, Optional, Dict

from termcolor import colored

from helper.data_structures import izip
from edit_distance.transposition_edit_distance import edit_operations as get_edit_operations, EditOperation
from evaluation.token_operations import Label, match_tokens, group_tokens, ground_truth_token_labels, \
    input_predicted_token_labels, input_true_token_labels, edit_labels2token_label
from tokenization.regex_tokenizer import RegexTokenizer
from evaluation.evaluated_sequence import TokenLabel, ErrorType, EvaluationCase, EvaluatedToken, EvaluatedSequence


def edit_operations(a: str, b: str, space_replace: bool) -> List[EditOperation]:
    return get_edit_operations(a, b, space_replace)


def all_in_set(elements: Iterable, set: Set) -> bool:
    n = 0
    for element in elements:
        n += 1
        if element not in set:
            return False
    if n == 0:
        return False
    return True


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
        correct_error_type = ErrorType.NONWORD if ErrorType.NONWORD in matched_corrupt_error_types \
            else ErrorType.REAL_WORD
        corrupt_error_types.extend(matched_corrupt_error_types)
        correct_error_types.extend([correct_error_type] * len(correct_indices))
    return correct_error_types, corrupt_error_types, matchings


def percentage_string(percentage):
    percentage = percentage * 100
    if percentage < 10:
        txt = "  "
    elif percentage < 100:
        txt = " "
    else:
        txt = ""
    return txt + "%.2f" % percentage


COLUMN_WIDTH = 20


def table_column(entry: str, width: int = COLUMN_WIDTH):
    if len(entry) < width:
        entry += " " * (width - len(entry))
    return entry


class Evaluator:
    ERROR_LABELS = [None] + sorted(ErrorType) + sorted([label for label in TokenLabel if label != TokenLabel.NONE])

    def __init__(self, words: Set[str], verbose: bool = False):
        self.count = {
            label: {
                case: 0 for case in EvaluationCase
            } for label in list(TokenLabel) + list(ErrorType)
        }
        self.words = words
        self.tokenizer = RegexTokenizer()
        self.n_sequences = 0
        self.n_correct_sequences = 0
        self.verbose = verbose

    def add(self, labels: List[Label], case: EvaluationCase, error_type: ErrorType):
        label = edit_labels2token_label(labels)
        self.count[label][case] += 1
        self.count[error_type][case] += 1

    def add_sequence_result(self, is_correct: bool):
        self.n_sequences += 1
        if is_correct:
            self.n_correct_sequences += 1

    def sequence_accuracy(self):
        return self.n_correct_sequences / self.n_sequences

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
        for label in Evaluator.ERROR_LABELS:
            counts = self._get_counts(label)
            correction_precision = fraction(counts[EvaluationCase.TRUE_POSITIVE],
                                            counts[EvaluationCase.TRUE_POSITIVE] + counts[
                                                EvaluationCase.FALSE_POSITIVE])
            correction_recall = fraction(counts[EvaluationCase.TRUE_POSITIVE],
                                         counts[EvaluationCase.TRUE_POSITIVE] + counts[EvaluationCase.FALSE_NEGATIVE])
            correction_f1 = f1(correction_precision, correction_recall)
            detection_precision = fraction(counts[EvaluationCase.DID_DETECT],
                                           counts[EvaluationCase.DID_DETECT] + counts[EvaluationCase.DAMAGED])
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
                print("%s | %s %s %s | %s %s %s" % (
                    table_column(label, width=11),
                    table_column("%s (%i/%i)" % (percentage_string(detection_precision),
                                                 counts[EvaluationCase.DID_DETECT],
                                                 counts[EvaluationCase.DID_DETECT]
                                                 + counts[EvaluationCase.DAMAGED])),
                    table_column("%s (%i/%i)" % (percentage_string(detection_recall),
                                                 counts[EvaluationCase.WAS_DETECTED],
                                                 counts[EvaluationCase.WAS_DETECTED]
                                                 + counts[EvaluationCase.UNDETECTED])),
                    percentage_string(detection_f1),
                    table_column("%s (%i/%i)" % (percentage_string(correction_precision),
                                                 counts[EvaluationCase.TRUE_POSITIVE],
                                                 counts[EvaluationCase.TRUE_POSITIVE]
                                                 + counts[EvaluationCase.FALSE_POSITIVE])),
                    table_column("%s (%i/%i)" % (percentage_string(correction_recall),
                                                 counts[EvaluationCase.TRUE_POSITIVE],
                                                 counts[EvaluationCase.TRUE_POSITIVE]
                                                 + counts[EvaluationCase.FALSE_NEGATIVE])),
                    percentage_string(correction_f1)
                ))

    def print_evaluation(self):
        header = "TYPE%s| DETECTION (precision, recall, F1)%s| CORRECTION (precision, recall, F1)" % (" " * 8, " " * 16)
        print(header)
        print("-" * 114)
        self.print_table(latex=False)
        print("-" * 114)

    def evaluate_sample(self,
                        correct: str,
                        corrupt: str,
                        predicted: str):
        evaluations = []

        correct_tokens = correct.split()
        corrupt_tokens = corrupt.split()
        predicted_tokens = predicted.split()

        # matchings
        pred_gt_matching = match_tokens(predicted_tokens, correct_tokens)
        pred_in_matching = match_tokens(predicted_tokens, corrupt_tokens)

        # gt labels
        gt_correctly_predicted = set(i for _, i in pred_gt_matching)
        gt_labels = ground_truth_token_labels(correct, corrupt)
        gt_not_corrupt = set(i for i, labels in enumerate(gt_labels) if len(labels) == 0)

        # in labels
        in_true_labels = input_true_token_labels(correct, corrupt)
        in_predicted_labels = input_predicted_token_labels(corrupt, predicted)

        # pred labels
        pred_correct = set(i for i, _ in pred_gt_matching)
        pred_unchanged = set(i for i, _ in pred_in_matching)

        # error types
        gt_error_types, in_error_types, gt2in_grouping = error_types(correct, corrupt, self.tokenizer, self.words)
        gt2inputs = index2group_matching(gt2in_grouping)

        # GROUND TRUTH SEQUENCE

        gt_sequence = ""
        gt_evaluated_tokens = []
        for gt_i, gt_token, labels, error_type in izip(correct_tokens, gt_labels, gt_error_types):
            color = None
            case = None
            if gt_i not in gt_not_corrupt:
                if gt_i in gt_correctly_predicted and any_edited(gt2inputs[gt_i], in_predicted_labels):
                    color = "green"
                    case = EvaluationCase.TRUE_POSITIVE
                else:
                    color = "yellow"
                    case = EvaluationCase.FALSE_NEGATIVE
                evaluations.append((labels, case, error_type))
            elif gt_i not in gt_correctly_predicted:
                color = "red"
                case = EvaluationCase.FALSE_POSITIVE
            if gt_i > 0:
                gt_sequence += ' '
            text = gt_token
            if len(labels) > 0:
                text += "[%s]" % edit_labels2token_label(labels).name
                text += "{%s}" % ERROR_TYPE_ABBREVIATIONS[error_type]
            gt_sequence += colored(text, color)
            # evaluated token:
            true_label = edit_labels2token_label(labels)
            eval_token = EvaluatedToken(gt_token, error_type, true_label, None, case)
            gt_evaluated_tokens.append(eval_token)

        # INPUT SEQUENCE

        in_correctly_predicted = set()
        for corrupt_group, predicted_group in group_tokens(corrupt, predicted):
            if all_in_set(predicted_group, pred_correct):
                for in_i in corrupt_group:
                    in_correctly_predicted.add(in_i)

        in_sequence = ""
        in_evaluated_tokens = []
        for in_i, in_token, true_labels, predicted_labels, error_type in \
                izip(corrupt_tokens, in_true_labels, in_predicted_labels, in_error_types):
            if in_i > 0:
                in_sequence += ' '
            text = in_token
            true_label = edit_labels2token_label(true_labels)
            predicted_label = edit_labels2token_label(predicted_labels)
            if len(predicted_labels) > 0 or len(true_labels) > 0:
                text += "[%s->%s]" % (true_label.name, predicted_label.name)
                text += "{%s}" % ERROR_TYPE_ABBREVIATIONS[error_type]
            color = None
            is_corrupt = len(true_labels) > 0
            changed = len(predicted_labels) > 0
            is_correct = in_i in in_correctly_predicted
            case = None
            if is_corrupt:
                if changed:
                    color = "green" if is_correct else "blue"
                    evaluations.append((predicted_labels, EvaluationCase.DID_DETECT, error_type))
                    evaluations.append((true_labels, EvaluationCase.WAS_DETECTED, error_type))
                    case = EvaluationCase.TRUE_POSITIVE if is_correct else EvaluationCase.WAS_DETECTED
                else:
                    color = "yellow"
                    evaluations.append((true_labels, EvaluationCase.UNDETECTED, error_type))
                    case = EvaluationCase.UNDETECTED
            else:
                if changed:
                    color = "red"
                    evaluations.append((predicted_labels, EvaluationCase.DAMAGED, error_type))
                    case = EvaluationCase.FALSE_POSITIVE
            if changed and not is_correct:
                evaluations.append((predicted_labels, EvaluationCase.FALSE_POSITIVE, error_type))
            in_sequence += colored(text, color)
            # evaluated token:
            eval_token = EvaluatedToken(in_token, error_type, true_label, predicted_label, case)
            in_evaluated_tokens.append(eval_token)

        # PREDICTED SEQUENCE

        pred_sequence = ""
        pred_evaluated_tokens = []
        for pred_i, pred_token in enumerate(predicted_tokens):
            color = None
            case = None
            if pred_i not in pred_unchanged and pred_i in pred_correct:
                color = "green"
                case = EvaluationCase.TRUE_POSITIVE
            elif pred_i in pred_unchanged and pred_i not in pred_correct:
                color = "yellow"
                case = EvaluationCase.FALSE_NEGATIVE
            elif pred_i not in pred_unchanged and pred_i not in pred_correct:
                color = "red"
                case = EvaluationCase.FALSE_POSITIVE
            if pred_i > 0:
                pred_sequence += ' '
            pred_sequence += colored(pred_token, color)
            # evaluated token:
            eval_token = EvaluatedToken(pred_token, None, None, None, case)
            pred_evaluated_tokens.append(eval_token)

        if self.verbose:
            print("GROUND TRUTH:\n" + gt_sequence)
            print("INPUT:\n" + in_sequence)
            print("PREDICTED:\n" + pred_sequence)
            print()

        evaluated_sequence = EvaluatedSequence(correct, corrupt, predicted, gt_evaluated_tokens, in_evaluated_tokens,
                                               pred_evaluated_tokens)

        is_correct = int(predicted == correct)

        return evaluations, is_correct, evaluated_sequence

    def get_results_dict(self):
        results_dict = {}
        for label in Evaluator.ERROR_LABELS:
            counts = self._get_counts(label)
            correction_n = counts[EvaluationCase.TRUE_POSITIVE] + counts[EvaluationCase.FALSE_NEGATIVE]
            correction_precision = fraction(counts[EvaluationCase.TRUE_POSITIVE],
                                            counts[EvaluationCase.TRUE_POSITIVE] + counts[
                                                EvaluationCase.FALSE_POSITIVE])
            correction_recall = fraction(counts[EvaluationCase.TRUE_POSITIVE], correction_n)
            correction_f1 = f1(correction_precision, correction_recall)
            detection_precision = fraction(counts[EvaluationCase.DID_DETECT],
                                           counts[EvaluationCase.DID_DETECT] + counts[EvaluationCase.DAMAGED])
            detection_recall = fraction(counts[EvaluationCase.WAS_DETECTED],
                                        counts[EvaluationCase.WAS_DETECTED] + counts[EvaluationCase.UNDETECTED])
            detection_f1 = f1(detection_precision, detection_recall)
            results_dict["all" if label is None else label.name] = {
                "correction": {
                    "precision": correction_precision,
                    "recall": correction_recall,
                    "f1": correction_f1
                },
                "detection": {
                    "precision": detection_precision,
                    "recall": detection_recall,
                    "f1": detection_f1
                },
                "n": correction_n
            }
        results_dict["accuracy"] = self.sequence_accuracy()
        results_dict["sequences"] = self.n_sequences
        return results_dict


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
