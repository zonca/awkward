# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def is_option(type_):
    return isinstance(type_, ak.types.OptionType)


def is_union(type_):
    return isinstance(type_, ak.types.UnionType)


def is_numpy(type_):
    return isinstance(type_, ak.types.NumpyType)


def is_unknown(type_):
    return isinstance(type_, ak.types.UnknownType)


def is_regular(type_):
    return isinstance(type_, ak.types.RegularType)


def is_list(type_):
    return isinstance(type_, ak.types.ListType)


def is_record(type_):
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


def determine_form_for_enforcing_type(form, type_, ctx):
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
        child = determine_form_for_enforcing_type(form.content, type_, ctx)
        return ctx.choose(
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
        return determine_form_for_enforcing_type(form.content, type_.content, ctx).copy(
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
        return determine_form_for_enforcing_type(form.content, type_.content, ctx).copy(
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

    elif form.is_union and is_union(type_):
        raise NotImplementedError
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
            f"{type(form).__name__} can only be converted into a UnionType if it is compatible with one "
            "of its contents, but no compatible content as found"
        )
    else:
        raise NonEnforcibleConversionError


def test():
    numpy_layout = ak.to_layout([1.0, 2.0, 3.0])
    numpy_form = numpy_layout.form

    # Change NumPy DType
    assert determine_form_for_enforcing_type(
        numpy_form, ak.types.from_datashape("int64", highlevel=False), Ctx()
    ) == ak.forms.NumpyForm("int64")

    # Unknown to type
    assert determine_form_for_enforcing_type(
        ak.forms.EmptyForm(), ak.types.from_datashape("int64", highlevel=False), Ctx()
    ) == ak.forms.NumpyForm("int64")

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
    # Option to option-of-unknown
    assert determine_form_for_enforcing_type(
        ak.forms.ByteMaskedForm("i8", ak.forms.NumpyForm("int32"), valid_when=True),
        ak.types.OptionType(ak.types.UnknownType()),
        Ctx(),
    ) == ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm())

    # Add option
    assert determine_form_for_enforcing_type(
        numpy_form, ak.types.from_datashape("?int64", highlevel=False), Ctx()
    ) == ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64"))

    # Remove (unmasked) option
    assert determine_form_for_enforcing_type(
        ak.forms.UnmaskedForm(numpy_form),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.NumpyForm("int64")

    # Remove (unmasked) option (introduce index)
    assert determine_form_for_enforcing_type(
        ak.forms.IndexedOptionForm("i64", numpy_form),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.IndexedForm("i64", ak.forms.NumpyForm("int64"))

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
    ) == ak.forms.RecordForm(
        [
            ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("int64")),
            ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float32")),
        ],
        ["x", "y"],
    )
