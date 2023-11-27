# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from itertools import permutations

import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward._parameters import type_parameters_equal
from awkward._typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from awkward.forms import Form
    from awkward.contents import Content
    from awkward.types import (
        ListType,
        NumpyType,
        OptionType,
        RecordType,
        RegularType,
        Type,
        UnionType,
        UnknownType,
    )


def is_option(type_: Type) -> TypeGuard[OptionType]:
    return isinstance(type_, ak.types.OptionType)


def is_union(type_: Type) -> TypeGuard[UnionType]:
    return isinstance(type_, ak.types.UnionType)


def is_numpy(type_: Type) -> TypeGuard[NumpyType]:
    return isinstance(type_, ak.types.NumpyType)


def is_unknown(type_: Type) -> TypeGuard[UnknownType]:
    return isinstance(type_, ak.types.UnknownType)


def is_regular(type_: Type) -> TypeGuard[RegularType]:
    return isinstance(type_, ak.types.RegularType)


def is_list(type_: Type) -> TypeGuard[ListType]:
    return isinstance(type_, ak.types.ListType)


def is_record(type_: Type) -> TypeGuard[RecordType]:
    return isinstance(type_, ak.types.RecordType)


class Ctx:
    def __init__(self, parent=None):
        self._is_set = False
        self._parent = parent

    def set(self):
        self._is_set = True
        if self._parent is not None:
            self._parent.set()

    @property
    def is_set(self):
        return self._is_set

    def child(self):
        return type(self)(self)

    def choose(self, on_set, on_unset):
        if self._is_set:
            return on_set
        else:
            return on_unset

    def __bool__(self):
        return self._is_set


class NonEnforcibleConversionError(TypeError):
    ...


def form_has_type(form: Form, type_: Type) -> bool:
    """
    Args:
        form: content object
        type_: low-level type object

    Returns True if the form satisfies the given type;, otherwise False.
    """
    if not type_parameters_equal(form._parameters, type_._parameters):
        return False

    if form.is_unknown:
        return isinstance(type_, ak.types.UnknownType)
    elif form.is_option:
        return isinstance(type_, ak.types.OptionType) and form_has_type(
            form.content, type_.content
        )
    elif form.is_indexed:
        return form_has_type(form.content, type_)
    elif form.is_regular:
        return (
            isinstance(type_, ak.types.RegularType)
            and (
                form.size is unknown_length
                or type_.size is unknown_length
                or form.size == type_.size
            )
            and form_has_type(form.content, type_.content)
        )
    elif form.is_list:
        return isinstance(type_, ak.types.ListType) and form_has_type(
            form.content, type_.content
        )
    elif form.is_numpy:
        for _ in range(form.purelist_depth - 1):
            if not isinstance(type_, ak.types.RegularType):
                return False
            type_ = type_.content
        return (
            isinstance(type_, ak.types.NumpyType) and form.primitive == type_.primitive
        )
    elif form.is_record:
        if (
            not isinstance(type_, ak.types.RecordType)
            or type_.is_tuple != form.is_tuple
        ):
            return False

        if form.is_tuple:
            return all(
                form_has_type(c, t) for c, t in zip(form.contents, type_.contents)
            )
        else:
            return (frozenset(form.fields) == frozenset(type_.fields)) and all(
                form_has_type(form.content(f), type_.content(f)) for f in type_.fields
            )
    elif form.is_union:
        if len(form.contents) != len(type_.contents):
            return False

        for contents in permutations(form.contents):
            if all(
                form_has_type(form, type_)
                for form, type_ in zip(contents, type_.contents)
            ):
                return True
        return False
    else:
        raise TypeError(form)


def determine_form_for_enforcing_type(form: Form, type_: Type, ctx: Ctx) -> Form:
    # Unknowns become canonical forms
    if form.is_unknown:
        return ak.forms.from_type(type_).copy(parameters=type_._parameters)
    # Unknown types must be children of indexed-option types
    elif is_unknown(type_):
        raise NonEnforcibleConversionError(
            "cannot convert non-option types to unknown types. "
            "To introduce an UnknownType, it must be wrapped in an OptionType"
        )
    # Option-of-unknown-type can be projected to
    elif form.is_option and is_option(type_) and is_unknown(type_.content):
        return ak.forms.IndexedOptionForm(
            "i64", ak.forms.EmptyForm(), parameters=type_._parameters
        )
    # Preserve option
    elif form.is_option and is_option(type_):
        child = determine_form_for_enforcing_type(form.content, type_.content, ctx)
        return ctx.choose(
            # If packed, pack content and build bytemask
            ak.forms.ByteMaskedForm("int8", child, True, parameters=type_._parameters),
            # Otherwise, preserve existing option
            form.copy(content=child, parameters=type_._parameters),
        )
    # Remove non-projecting option
    elif isinstance(form, ak.forms.UnmaskedForm) and not is_option(type_):
        return determine_form_for_enforcing_type(form.content, type_, ctx)
    # Remove option
    elif form.is_option and not is_option(type_):
        child_ctx = ctx.child()
        # We need to project out the option
        ctx.set()
        # But the child context does not, because we can introduce an indexed node
        child = determine_form_for_enforcing_type(form.content, type_, child_ctx)
        return child_ctx.choose(
            # If packed, drop option node entirely
            child,
            # Else, use an indexed node
            ak.forms.IndexedForm(getattr(form, "index", "i64"), child),
        )
    # Add option
    elif not form.is_option and is_option(type_):
        return ak.forms.UnmaskedForm.simplified(
            determine_form_for_enforcing_type(form, type_.content, ctx),
            parameters=type_._parameters,
        )
    # Keep or project indexed node? Determine by later conversion.
    elif form.is_indexed:
        child = determine_form_for_enforcing_type(form.content, type_, ctx)
        return ctx.choose(
            child,
            form.copy(content=child),
        )
    # Regular to regular
    elif form.is_regular and is_regular(type_):
        # regular → regular requires same size!
        if form.size != type_.size:
            raise NonEnforcibleConversionError(
                f"regular form has different size ({form.size}) to type ({type_.size})"
            )
        return form.copy(content=determine_form_for_enforcing_type(form.content, type_.content, ctx),
            parameters=type_._parameters
        )
    # Regular to list
    elif form.is_regular and is_list(type_):
        return ak.forms.ListOffsetForm(
            "i64",
            determine_form_for_enforcing_type(form.content, type_.content, ctx),
            parameters=type_.parameters,
        )
    # List to list
    elif form.is_list and is_list(type_):
        return form.copy(
            content=determine_form_for_enforcing_type(form.content, type_.content, ctx),
            parameters=type_._parameters
        )
    # Change dtype!
    elif form.is_numpy and form.inner_shape == () and is_numpy(type_):
        return ak.forms.NumpyForm(type_.primitive, parameters=type_._parameters)
    # N-D NumpyForms should be regularised
    elif form.is_numpy and len(form.inner_shape) >= 1:
        return determine_form_for_enforcing_type(form.to_RegularForm(), type_, ctx)
    # Record → Record
    elif (
        form.is_record and not form.is_tuple and is_record(type_) and not type_.is_tuple
    ):
        form_fields = frozenset(form._fields)

        # Compute existing and new fields
        # Use lists to preserve type order
        existing_fields = []
        new_fields = []
        for field in type_.fields:
            if field in form_fields:
                existing_fields.append(field)
            else:
                new_fields.append(field)
        next_fields = existing_fields + new_fields
        # Recurse into shared contents
        next_contents = [
            determine_form_for_enforcing_type(form.content(f), type_.content(f), ctx)
            for f in existing_fields
        ]
        for field in new_fields:
            field_type = type_.content(field)

            # Added types must be options, so that they can be all-None
            if not is_option(field_type):
                raise NonEnforcibleConversionError(
                    "can only add new fields to a record if they are option types"
                )
            # Append new contents
            next_contents.append(
                ak.forms.IndexedOptionForm(
                    getattr(form, "index", "i64"),
                    ak.forms.from_type(field_type.content),
                )
            )

        return form.copy(
            fields=next_fields,
            contents=next_contents,
            parameters=type_._parameters,
        )
    # Tuple → Tuple
    elif form.is_record and form.is_tuple and is_record(type_) and type_.is_tuple:
        type_contents = iter(type_.contents)
        next_contents = [
            determine_form_for_enforcing_type(c, t, ctx)
            for c, t in zip(form.contents, type_contents)
        ]
        for next_type in type_contents:
            if not is_option(next_type):
                raise NonEnforcibleConversionError(
                    "can only add new slots to a tuple if they are option types"
                )
            next_contents.append(
                ak.forms.IndexedOptionForm(
                    getattr(form, "index", "i64"), ak.forms.from_type(next_type.content)
                )
            )
        return form.copy(
            fields=None, contents=next_contents, parameters=type_.parameters
        )
    # Record/Tuple → Tuple/Record
    elif form.is_record and is_record(type_):
        raise NonEnforcibleConversionError(
            "records and tuples cannot be converted between one another"
        )
    # Union projection
    elif form.is_union and not is_union(type_):
        # If all contents are enforcible, return the canonical type (easy)
        try:
            next_contents = [
                determine_form_for_enforcing_type(content, type_, ctx)
                for content in form.contents
            ]

        # Otherwise, look for a projection
        except NonEnforcibleConversionError:
            # Return canonical type if we can project
            for content in form.contents:
                if content.type.is_equal_to(type_):
                    return content.copy(parameters=type_._parameters)
            else:
                raise NonEnforcibleConversionError(
                    f"UnionForm(s) can only be converted into {type_} if it has the same type, "
                    f"but no content with type {type_} was found"
                )
        else:
            # Perhaps should use dedicated `mergemany` implementation for forms?
            # Or, could return "least expensive" type e.g. `forms.from_type`?
            return ak.forms.UnionForm.simplified(
                "i8", "i64", next_contents, parameters=type_._parameters
            )

    # Up to one changed content
    elif (
        form.is_union and is_union(type_) and len(type_.contents) == len(form.contents)
    ):
        # Type and form have same number of contents. Up-to *one* content can differ
        for permuted_types in permutations(type_.contents):
            content_matches_type = [
                form_has_type(content, permuted_type)
                for content, permuted_type in zip(form.contents, permuted_types)
            ]
            n_matching = sum(content_matches_type)

            if n_matching >= len(type_.contents) - 1:
                return form.copy(
                    contents=[
                        determine_form_for_enforcing_type(content, permuted_type, ctx)
                        for content, permuted_type in zip(form.contents, permuted_types)
                    ],
                    parameters=type_._parameters,
                )
            else:
                raise TypeError(
                    "UnionArray(s) can currently only be converted into UnionArray(s) with the same number of contents "
                    "if no greater than one content differs in type"
                )

    # Add new content
    elif form.is_union and is_union(type_) and len(type_.contents) > len(form.contents):
        for permuted_types in permutations(type_.contents, len(form.contents)):
            if all(
                form_has_type(content, permuted_type)
                for content, permuted_type in zip(form.contents, permuted_types)
            ):
                contents = [
                    determine_form_for_enforcing_type(content, permuted_type, ctx)
                    for content, permuted_type in zip(form.contents, permuted_types)
                ]
                contents.extend(
                    [
                        ak.forms.from_type(t)
                        for t in type_.contents
                        if t not in permuted_types
                    ]
                )
                return form.copy(contents=contents, parameters=type_._parameters)
            else:
                raise TypeError(
                    "UnionForm can currently only be converted into UnionType with a greater "
                    "number of contents if the UnionForm's contents are compatible with some permutation of "
                    "the UnionType's contents"
                )
    # Drop content
    elif form.is_union and is_union(type_):
        for permuted_contents in permutations(form.contents, len(type_.contents)):
            if all(
                form_has_type(permuted_content, positional_type)
                for positional_type, permuted_content in zip(
                    type_.contents, permuted_contents
                )
            ):
                contents = [
                    determine_form_for_enforcing_type(
                        permuted_content, positional_type, ctx
                    )
                    for positional_type, permuted_content in zip(
                        type_.contents, permuted_contents
                    )
                ]

                # We need to project out unused items, so that the content
                # can be dropped
                ctx.set()
                return form.copy(contents=contents, parameters=type_._parameters)
            else:
                raise TypeError(
                    "UnionForm can currently only be converted into UnionType with fewer "
                    "contents if the UnionType's contents are compatible with some permutation of "
                    "the UnionForm's contents"
                )

    # Add a union!
    elif is_union(type_):
        current_type = form.type
        for i, content_type in enumerate(type_.contents):
            if not current_type.is_equal_to(content_type):
                continue

            other_contents = [
                ak.forms.from_type(t) for j, t in enumerate(type_.contents) if j != i
            ]

            return ak.forms.UnionForm(
                tags="i8",
                index="i64",
                contents=[
                    determine_form_for_enforcing_type(form, content_type, ctx),
                    *other_contents,
                ],
                parameters=type_._parameters,
            )

        raise NonEnforcibleConversionError(
            "UnionForm can only be converted into a UnionType if it is compatible with one "
            "of its contents, but no compatible content as found"
        )
    else:
        raise NonEnforcibleConversionError


def enforce_form(layout: Content, form: Form, ctx: Ctx) -> Content:
    if layout.is_numpy and form.is_numpy:



def test():
    # Change NumPy DType
    assert determine_form_for_enforcing_type(
        ak.forms.NumpyForm("int64"),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.NumpyForm("int64")


def test_unknown_to_any():
    # Unknown to type
    assert determine_form_for_enforcing_type(
        ak.forms.EmptyForm(), ak.types.from_datashape("int64", highlevel=False), Ctx()
    ) == ak.forms.NumpyForm("int64")


def test_any_to_unknown():
    # Form to unknown
    with pytest.raises(
        NonEnforcibleConversionError,
        match="cannot convert non-option types to unknown types",
    ):
        assert determine_form_for_enforcing_type(
            ak.forms.NumpyForm("int32"),
            ak.types.OptionType(ak.types.UnknownType()),
            Ctx(),
        )


def test_option_to_option_unknown():
    # Option to option-of-unknown
    assert determine_form_for_enforcing_type(
        ak.forms.ByteMaskedForm("i8", ak.forms.NumpyForm("int32"), valid_when=True),
        ak.types.OptionType(ak.types.UnknownType()),
        Ctx(),
    ) == ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm())


def test_non_option_to_option():
    # Add option
    assert determine_form_for_enforcing_type(
        ak.forms.NumpyForm("int64"),
        ak.types.from_datashape("?int64", highlevel=False),
        Ctx(),
    ) == ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64"))


def test_option_to_non_option_unmasked():
    # Remove (unmasked) option
    assert determine_form_for_enforcing_type(
        ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64")),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.NumpyForm("int64")


def test_option_to_non_option_no_projection():
    # Remove (indexed) option (introduce index)
    assert determine_form_for_enforcing_type(
        ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("int64")),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.IndexedForm("i64", ak.forms.NumpyForm("int64"))


def test_option_to_non_option_projection():
    # Remove (indexed) option (don't introduce index)
    # The inner indexed node is still preserved
    assert determine_form_for_enforcing_type(
        ak.forms.IndexedOptionForm(
            "i64", ak.forms.ListOffsetForm("i64", ak.forms.IndexedOptionForm("i32", ak.forms.NumpyForm("int64")))
        ),
        ak.types.from_datashape("var * int64", highlevel=False),
        Ctx(),
    ) == ak.forms.ListOffsetForm("i64", ak.forms.IndexedForm("i32", ak.forms.NumpyForm("int64")))


def test_add_record_field():
    # Add new field
    assert determine_form_for_enforcing_type(
        ak.forms.RecordForm([ak.forms.NumpyForm("int64")], fields=("x",)),
        ak.types.from_datashape("{x: int64, y: ?float32}", highlevel=False),
        Ctx(),
    ) == ak.forms.RecordForm(
        [
            ak.forms.NumpyForm("int64"),
            ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float32")),
        ],
        fields=("x", "y"),
    )


def test_add_record_slot():
    # Add new slot
    assert determine_form_for_enforcing_type(
        ak.forms.RecordForm([ak.forms.NumpyForm("int64")], fields=None),
        ak.types.from_datashape("(int64, ?float32)", highlevel=False),
        Ctx(),
    ) == ak.forms.RecordForm(
        [
            ak.forms.NumpyForm("int64"),
            ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float32")),
        ],
        fields=None,
    )


def test_adda_and_remove_record_field():
    # Remove field, add new field
    assert determine_form_for_enforcing_type(
        ak.forms.RecordForm([ak.forms.NumpyForm("int64")], fields=("x",)),
        ak.types.from_datashape("{y: ?int64}", highlevel=False),
        Ctx(),
    ) == ak.forms.RecordForm(
        [
            ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("int64")),
        ],
        fields=("y",),
    )


def test_change_record_slot():
    # Change slot
    assert determine_form_for_enforcing_type(
        ak.forms.RecordForm([ak.forms.NumpyForm("int64")], fields=None),
        ak.types.from_datashape("(?float32)", highlevel=False),
        Ctx(),
    ) == ak.forms.RecordForm(
        [
            ak.forms.UnmaskedForm(ak.forms.NumpyForm("float32")),
        ],
        fields=None,
    )


def test_tuple_to_record():
    # Change tuple to record
    with pytest.raises(
        NonEnforcibleConversionError,
        match="records and tuples cannot be converted between one another",
    ):
        determine_form_for_enforcing_type(
            ak.forms.RecordForm([ak.forms.NumpyForm("int64")], fields=None),
            ak.types.from_datashape("{x: float32}", highlevel=False),
            Ctx(),
        )


def test_record_to_tuple():
    # Change record to tuple
    with pytest.raises(
        NonEnforcibleConversionError,
        match="records and tuples cannot be converted between one another",
    ):
        determine_form_for_enforcing_type(
            ak.forms.RecordForm([ak.forms.NumpyForm("int64")], fields=("x",)),
            ak.types.from_datashape("(float32)", highlevel=False),
            Ctx(),
        )


def test_non_union_to_union():
    # Create union
    assert determine_form_for_enforcing_type(
        ak.forms.NumpyForm("int64"),
        ak.types.from_datashape("union[int64, datetime64]", highlevel=False),
        Ctx(),
    ) == ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.NumpyForm("int64"),
            ak.forms.NumpyForm("datetime64"),
        ],
    )


def test_project_union():
    # Project single content
    assert determine_form_for_enforcing_type(
        ak.forms.UnionForm(
            "i8",
            "i64",
            [
                ak.forms.NumpyForm("int64"),
                ak.forms.RecordForm([ak.forms.NumpyForm("float32")], ["y"]),
            ],
        ),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.NumpyForm("int64")

    # Enforce all contents
    assert determine_form_for_enforcing_type(
        ak.forms.UnionForm(
            "i8",
            "i64",
            [
                ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
                ak.forms.RecordForm([ak.forms.NumpyForm("float32")], ["y"]),
            ],
        ),
        ak.types.from_datashape("{x: ?int64, y: ?float32}", highlevel=False),
        Ctx(),
    ) == ak.forms.IndexedForm(
        # This form adds an outer index because of the carry of a record array
        "i64",
        ak.forms.RecordForm(
            [
                ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("int64")),
                ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float32")),
            ],
            ["x", "y"],
        ),
    )


def test_drop_union_contents():
    # Enforce fewer contents
    assert determine_form_for_enforcing_type(
        ak.forms.UnionForm(
            "i8",
            "i64",
            [
                ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
                ak.forms.RecordForm([ak.forms.NumpyForm("float32")], ["y"]),
                ak.forms.RecordForm([ak.forms.NumpyForm("datetime64")], ["z"]),
            ],
        ),
        ak.types.from_datashape("union[{x: int64}, {y: float32}]", highlevel=False),
        Ctx(),
    ) == ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
            ak.forms.RecordForm([ak.forms.NumpyForm("float32")], ["y"]),
        ],
    )


def test_expand_union_contents():
    # Enforce greater contents
    assert determine_form_for_enforcing_type(
        ak.forms.UnionForm(
            "i8",
            "i64",
            [
                ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
                ak.forms.RecordForm([ak.forms.NumpyForm("float32")], ["y"]),
            ],
        ),
        ak.types.from_datashape(
            "union[{x: int64}, {y: float32}, {z: ?datetime64[s]}]", highlevel=False
        ),
        Ctx(),
    ) == ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
            ak.forms.RecordForm([ak.forms.NumpyForm("float32")], ["y"]),
            ak.forms.RecordForm(
                [
                    ak.forms.IndexedOptionForm(
                        "i64", ak.forms.NumpyForm("datetime64[s]")
                    )
                ],
                ["z"],
            ),
        ],
    )
