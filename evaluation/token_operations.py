from typing import Union, Tuple, Optional, List

from enum import Enum

from edit_distance.transposition_edit_distance import edit_operations, EditOperation, OperationType
from helper.data_structures import izip


class TokenOperationType(Enum):
    MERGE = 0
    SPLIT = 1
    REPLACE_SINGLE = 2
    REPLACE_MULTI = 3

    def __lt__(self, other):
        return self.value < other.value


class TokenOperation:
    def __init__(self,
                 type: TokenOperationType,
                 token_indices: Union[int, Tuple[int, ...]],
                 subtoken_index: Optional[int]=None,
                 split_position: Optional[int]=None,
                 word: Optional[str]=None,
                 edit_distance: Optional[int]=None):
        self.type = type
        self.token_indices = token_indices
        self.subtoken_index = subtoken_index
        self.split_position = split_position
        self.word = word
        self.edit_distance = edit_distance

    def __str__(self):
        return "TokenOperation(type=%s, tokens=%s, subtoken=%s, split=%s, word=%s, ed=%s)" % (
            self.type.name, str(self.token_indices), str(self.subtoken_index), str(self.split_position), str(self.word), str(self.edit_distance)
        )

    def _tuple(self):
        return self.type, self.token_indices, self.subtoken_index, self.split_position, self.word
        #return self.token_indices, self.subtoken_index, self.split_position

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
        return "ToḱenGroup(%i, %s, %s, %s)" % (self.start_pos, str(self.token_indices), str(self.split_positions), str(self.subgroup_edits))

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
