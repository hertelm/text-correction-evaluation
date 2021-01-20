from typing import Union, Tuple, Optional, List

import numpy as np

from helper.sortable_enum import SortableEnum
from edit_distance.transposition_edit_distance import edit_operations, EditOperation, OperationType
from helper.data_structures import izip
from evaluation.evaluated_sequence import TokenLabel


class TokenOperationType(SortableEnum):
    MERGE = 0
    SPLIT = 1
    REPLACE_SINGLE = 2
    REPLACE_MULTI = 3


class TokenOperation:
    def __init__(self,
                 type: TokenOperationType,
                 token_indices: Union[int, Tuple[int, ...]],
                 subtoken_index: Optional[int] = None,
                 split_position: Optional[int] = None,
                 word: Optional[str] = None,
                 edit_distance: Optional[int] = None):
        self.type = type
        self.token_indices = token_indices
        self.subtoken_index = subtoken_index
        self.split_position = split_position
        self.word = word
        self.edit_distance = edit_distance

    def __str__(self):
        return "TokenOperation(type=%s, tokens=%s, subtoken=%s, split=%s, word=%s, ed=%s)" % (
            self.type.name, str(self.token_indices), str(self.subtoken_index), str(self.split_position), str(self.word),
            str(self.edit_distance)
        )

    def _tuple(self):
        return self.type, self.token_indices, self.subtoken_index, self.split_position, self.word

    def __eq__(self, other):
        return self._tuple() == other._tuple()

    def __hash__(self):
        return hash(self._tuple())

    def _first_token(self):
        if isinstance(self.token_indices, int):
            return self.token_indices
        return self.token_indices[0]

    def _subtoken_index(self):
        return 0 if self.subtoken_index is None else self.subtoken_index

    def _split_position(self):
        return 0 if self.split_position is None else self.split_position

    def __lt__(self, other):
        if self._first_token() != other._first_token():
            return self._first_token() < other._first_token()
        if self._subtoken_index() != other._subtoken_index():
            return self._subtoken_index() < other._subtoken_index()
        if self.type != other.type:
            return self.type < other.type
        return self._split_position() < other._split_position()


def assign_edits_to_tokens(edits: List[EditOperation],
                           tokens: List[str]) -> List[List[EditOperation]]:
    e_i = 0
    pos = 0
    token_edits = []
    for t_i, token in enumerate(tokens):
        token_edits.append([])
        pos += len(token) + 1
        while e_i < len(edits) and edits[e_i].position < pos:
            token_edits[-1].append(edits[e_i])
            e_i += 1
    return token_edits


def is_merge_edit(operation: EditOperation):
    return operation.type == OperationType.DELETION and operation.character == ' '


def is_split_edit(operation: EditOperation):
    return operation.type == OperationType.INSERTION and operation.character == ' '


class TokenGroup:
    def __init__(self,
                 start_pos: int,
                 token_indices: Tuple[int, ...]):
        self.start_pos = start_pos
        self.token_indices = token_indices
        self.split_positions = []
        self.subgroup_edits = [[]]

    def __str__(self):
        return "ToḱenGroup(%i, %s, %s, %s)" % \
               (self.start_pos, str(self.token_indices), str(self.split_positions), str(self.subgroup_edits))

    def add_split(self, subtoken_index: int, sequence_position: int):
        relative_position = sequence_position - self.start_pos - len(self.split_positions) - subtoken_index
        self.split_positions.append(relative_position)
        self.subgroup_edits.append([])

    def add_edit(self, subtoken_index: int, edit: EditOperation):
        relative_split_pos = 0 if len(self.split_positions) == 0 else self.split_positions[-1]
        relative_pos = edit.position - self.start_pos - relative_split_pos - subtoken_index
        self.subgroup_edits[-1].append(
            EditOperation(
                type=edit.type,
                position=relative_pos,
                character=edit.character
            )
        )


def group_merges(tokens: List[str],
                 token_edits: List[List[EditOperation]]) -> List[TokenGroup]:
    merged_groups = []
    group_indices = []
    pos = 0
    start_pos = 0
    for t_i, token, edits in izip(tokens, token_edits):
        group_indices.append(t_i)
        pos += len(token) + 1
        if len(edits) == 0 or not is_merge_edit(edits[-1]):
            merged_groups.append(
                TokenGroup(start_pos=start_pos,
                           token_indices=tuple(group_indices))
            )
            group_indices = []
            start_pos = pos
    return merged_groups


def add_subgroups(groups: List[TokenGroup],
                  token_edits: List[List[EditOperation]]):
    for group in groups:
        for sub_i, t_i in enumerate(group.token_indices):
            for edit in token_edits[t_i]:
                if is_merge_edit(edit):
                    pass
                elif is_split_edit(edit):
                    group.add_split(sub_i, edit.position)
                else:
                    group.add_edit(sub_i, edit)


def token_groups2token_operations(token_groups: List[TokenGroup],
                                  result_tokens: List[str]) -> List[TokenOperation]:
    token_operations = []
    result_t_i = 0
    for group in token_groups:
        # merges:
        for left, right in zip(group.token_indices[:-1], group.token_indices[1:]):
            token_operations.append(
                TokenOperation(TokenOperationType.MERGE, (left, right))
            )
        # splits and replacements:
        for sub_i in range(len(group.subgroup_edits)):
            # split:
            if sub_i > 0:
                split_pos = group.split_positions[sub_i - 1]
                token_operations.append(
                    TokenOperation(TokenOperationType.SPLIT, group.token_indices, None, split_pos)
                )
            # replace:
            n_sub_edits = len(group.subgroup_edits[sub_i])
            if n_sub_edits > 0:
                word = result_tokens[result_t_i]
                type = TokenOperationType.REPLACE_SINGLE if n_sub_edits == 1 else TokenOperationType.REPLACE_MULTI
                token_operations.append(
                    TokenOperation(type, group.token_indices, sub_i, None, word, edit_distance=n_sub_edits)
                )
            result_t_i += 1
    return token_operations


def get_token_operations(a: str, b: str) -> List[TokenOperation]:
    edits = edit_operations(a, b, space_replace=False)
    a_tokens = a.split(' ')
    b_tokens = b.split(' ')
    token_edits = assign_edits_to_tokens(edits, a_tokens)
    merged_groups = group_merges(a_tokens, token_edits)
    add_subgroups(merged_groups, token_edits)
    token_operations = token_groups2token_operations(merged_groups, b_tokens)
    return token_operations


class Label(SortableEnum):
    EDIT = 0
    MERGE = 1
    SPLIT = 2


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
