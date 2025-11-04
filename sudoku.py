from __future__ import annotations
from functools import partial
from itertools import combinations, permutations
from operator import indexOf
from typing import Any, Generator, Literal
from unittest.mock import NonCallableMagicMock

from matplotlib.pyplot import box
import numpy as np
from numpy._typing import NDArray

from get_sdk import get_sdk


class Bitmask(np.ndarray):
    def __new__(cls, sdk: Sudoku):
        obj = super().__new__(cls, shape=sdk.board.shape, dtype=np.int64)

        obj.sdk = sdk

        index_iter = np.nditer(sdk.board, flags=["multi_index"])
        for char in index_iter:
            index = index_iter.multi_index
            char = int(char)  # type: ignore
            obj.update(index, char)

        return obj

    def __array_finalize__(self, obj: None | np.ndarray) -> None:
        if obj is None:
            return
        
        self.sdk: Sudoku = getattr(obj, "sdk")

    def update(self, index: tuple[int, ...], char: Any):
        if char == self.sdk.empty_char:
            return  # no information

        char_bitmask = self.sdk.char_bitmask_map[char]

        for ca in self.sdk.get_CAs_at(index):
            # essentially ~char_bitmask, but python is weird
            self[ca] &= self.sdk.full_bitmask ^ char_bitmask

        self[index] = 0

    def verify(self):
        # possibilities at a known location
        if np.any(self[np.where(self.sdk.board != self.sdk.empty_char)] != 0):
            return False

        # no possibilities at an unknown location
        if np.any(self[np.where(self.sdk.board == self.sdk.empty_char)] == 0):
            return False

        index_iterator = np.nditer(self.sdk.board, flags=["multi_index"])
        for char in index_iterator:
            index = index_iterator.multi_index
            char = int(char)  # type: ignore  # char is a 0d array

            if char == self.sdk.empty_char:
                continue

            char_bitmask = self.sdk.char_bitmask_map[char]

            for ca in self.sdk.get_CAs_at(index):
                if np.any((self[ca] & char_bitmask) != 0):
                    return False
        return True

    def compare_to(self, old: Bitmask) -> None:
        print(f"old is valid:    {old.verify()}")
        print(f"new is valid:    {self.verify()}")
        print()
        # compares
        n_delta = np.sum(old != self)
        print(
            f"delta indicies:  {n_delta} ({round(n_delta / (self.sdk.length ** self.sdk.ndim) * 100, 2)}%)"
        )

        old_n_pbits = np.sum(np.bitwise_count(old))
        print(f"total pbits old: {old_n_pbits}")

        new_n_pbits = np.sum(np.bitwise_count(self))
        print(f"total pbits new: {new_n_pbits}")

        # equivalent to old_n_pbits - new_n_pbits, this just does it per cell then adds it up
        # instead of doing it all at once then comparing
        removed_p = np.sum(np.bitwise_count(old & ~self))
        print(
            f"# deleted pbits: {removed_p} ({round(removed_p / np.sum(np.bitwise_count(old)) * 100, 2)}%)"
        )
        print()

        ns_old, ns_new = old.get_n_singles(), self.get_n_singles()
        print(f"old singles:     {ns_old}")
        print(f"new singles:     {ns_new} (+{ns_new - ns_old})")
    
    def get_n_singles(self):
        return (np.bitwise_count(self) == 1).sum()
    
    def reduce(self, min: int = 2, max: int = 7):
        for n in range(min, max):
            self.reduce_all_sets(n)

        for box_ca in self.sdk.get_all_box_CAs():
            for char_bitmask in self.sdk.unique_bits_array:
                self._reduce_pointing_sets(box_ca=box_ca, char_bitmask=char_bitmask)

    def reduce_all_sets(self, n: int):
        chars_bitmasks = list(self.sdk.get_char_bitmask_combinations(n))

        for ca in self.sdk.get_all_CAs():
            for chars_bitmask in chars_bitmasks:
                self._reduce_obvious_sets(ca, chars_bitmask, n)
                self._reduce_hidden_sets(ca, chars_bitmask, n)


    def _reduce_obvious_sets(self, ca: tuple, chars_bitmask: int, n: int) -> None:
        intersection = self[ca] & (self.sdk.full_bitmask ^ chars_bitmask)

        subset_mask = np.bitwise_count(intersection) == 0

        subset_mask &= self.sdk.board[ca] != self.sdk.empty_char

        if subset_mask.sum() != n:
            return
        
        self[ca][~subset_mask] &= self.sdk.full_bitmask ^ chars_bitmask

    def _reduce_hidden_sets(self, ca: tuple, chars_bitmask: int, n: int) -> None:
        superset_mask = np.bitwise_count(self[ca] & chars_bitmask) > 0

        if superset_mask.sum() != n:
            return

        self[ca][superset_mask] = chars_bitmask

    def _reduce_pointing_sets(self, box_ca: tuple, char_bitmask: int) -> None:
        if self.sdk.ca_type(box_ca) != "box":
            raise ValueError

        ...


class Sudoku:
    def __init__(
        self,
        board: np.ndarray,
        /,
        *,
        chars: np.ndarray | None = None,
        empty_char: Any | None = None,
        box_length: int = 3,
    ):
        self.board = board
        self.chars = chars or np.arange(1, 10)
        self.empty_char = empty_char or np.int8(0)
        self.box_length = box_length

        self.n_chars = len(self.chars)
        self.length = self.n_chars
        self.ndim = self.board.ndim

        self.bitmask_dtype = np.min_scalar_type(1 << self.n_chars)
        self.unique_bits_array: np.ndarray = 1 << np.arange(
            self.n_chars, dtype=self.bitmask_dtype
        )
        self.full_bitmask: int = (1 << self.n_chars) - 1
        self.char_bitmask_map = {
            char: self.unique_bits_array[np.where(self.chars == char)][0]
            for char in self.chars
        }

        self.bitmask = Bitmask(self)

    def get_CAs_at(self, index: tuple[int, ...]) -> Generator[tuple]:
        for axis_n in np.arange(self.ndim):
            yield (*index[:axis_n], slice(None), *index[axis_n + 1 :])
        yield self.get_box_CA_at(index)

    def get_box_CA_at(self, index: tuple[int, ...]) -> tuple:
        index_arr = np.asarray(index)
        box_start = index_arr - index_arr % self.box_length
        box_end = box_start + self.box_length
        as_slices = map(slice, box_start, box_end)
        return tuple(as_slices)

    def get_all_CAs(self) -> Generator[tuple]:
        for axis_n in range(self.ndim):
            other_axis_permutations = permutations(range(self.length), self.ndim - 1)
            for other_axes in other_axis_permutations:
                yield (*other_axes[:axis_n], slice(None), *other_axes[axis_n:])

        yield from self.get_all_box_CAs()

    def get_all_box_CAs(self):
        per_axis_steps = range(0, self.length, self.box_length)
        for box_start in permutations(per_axis_steps, self.ndim):
            yield self.get_box_CA_at(box_start)

        # permutations doesn't include indentical values
        for i in per_axis_steps:
            yield self.get_box_CA_at((i,) * self.ndim)

    def get_char_bitmask_combinations(self, n: int):
        yield from (
            self.merge_bitmasks(*masks)
            for masks in combinations(self.unique_bits_array, n)
        )

    def merge_bitmasks(self, *masks: int):
        base = 0
        for mask in masks:
            base |= mask
        return base

    def create_axis_CA(self, axis_n: int, value: int):
        return tuple(
            slice(None) if iter_axis_n != axis_n else value
            for iter_axis_n in range(self.ndim)
        )

    def get_containing_box(self, *indices: tuple[int, ...]) -> tuple | None:
        if not indices:
            return None

        box = self.get_box_CA_at(indices[0])

        if any(box != self.get_box_CA_at(i) for i in indices):
            return None

        return box

    def ca_type(self, ca: tuple) -> Literal["axis", "box"]:
        if all(type(i) is slice for i in ca):
            return "box"
        return "axis"

    def exhaust_singles(self) -> None:

        old: np.ndarray | None = None
        while not np.array_equal(old, self.bitmask):  # type: ignore
            old = self.bitmask.copy()
            self.resolve_singles()

    def resolve_singles(self) -> None:
        while True:
            # recompile after every iteration
            singles = list(zip(*np.where(np.bitwise_count(self.bitmask) == 1)))

            if not singles:
                return
            
            for index in singles:
                # must convert to index to use update_bitmask_with_value
                if np.bitwise_count(self.bitmask[index]) != 1:
                    continue

                char_mask = int(self.bitmask[index])
                if char_mask == 0:
                    continue

                char_index = np.log2(char_mask).astype(int)
                char = self.chars[char_index]

                self.board[index] = char

                self.bitmask.update(index, char)


    def solve(self):
        old = None

        while not np.array_equal(old, self.board):  # type: ignore
            old = self.board.copy()

            self.exhaust_singles()
            self.bitmask.reduce()
            self.exhaust_singles()

    def is_solved(self):
        if np.any(self.board == self.empty_char):
            return False

        if self.bitmask.any():
            return False

        for ca in self.get_all_CAs():
            if len(np.unique_values(self.board[ca])) != len(self.chars):
                return False

        return True

    def __repr__(self):
        return str(self.board).replace("0", " ")


if __name__ == "__main__":
    response = get_sdk("easy")

    sdk = Sudoku(response.board)
    print(sdk)

    sdk.solve()
    print(sdk)
    print(sdk.is_solved())
