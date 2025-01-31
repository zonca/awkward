# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import sys
from collections.abc import Mapping

import awkward as ak
from awkward._typing import Any
from awkward.types.type import Type


class ScalarType:
    def __init__(self, content: Type, behavior: Mapping | None = None):
        if not isinstance(content, ak.types.Type):
            raise TypeError(
                "{} all 'contents' must be Type subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        self._content: Type = content
        self._behavior: Mapping | None = behavior

    @property
    def content(self) -> Type:
        return self._content

    @property
    def behavior(self) -> Mapping | None:
        return self._behavior

    def __str__(self) -> str:
        return "".join(self._str("", True))

    def show(self, stream=sys.stdout):
        stream.write("".join([*self._str("", False), "\n"]))

    def _str(self, indent: str, compact: bool) -> list[str]:
        return self._content._str(
            indent,
            compact,
            self._behavior,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._content!r}, {self._behavior!r})"

    def is_equal_to(self, other: Any, *, all_parameters: bool = False) -> bool:
        return isinstance(other, type(self)) and self._content.is_equal_to(
            other._content, all_parameters=all_parameters
        )

    __eq__ = is_equal_to
