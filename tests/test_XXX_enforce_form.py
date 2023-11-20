# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak
from awkward._parameters import parameters_union


def test():
    def is_option(type_):
        return isinstance(type_, ak.types.OptionType)

    def is_numpy(type_):
        return isinstance(type_, ak.types.NumpyType)

    def is_unknown(type_):
        return isinstance(type_, ak.types.UnknownType)

    def is_regular(type_):
        return isinstance(type_, ak.types.RegularType)

    def is_list(type_):
        return isinstance(type_, ak.types.ListType)

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

    def determine_form_for_enforcing_type(form, type_, ctx):
        # Unknowns become canonincal forms
        if form.is_unknown:
            return ak.forms.from_type(type_)
        elif is_unknown(type_):
            raise TypeError(
                "cannot convert non-EmptyArray layouts to a bare UnknownType. "
                "To introduce an UnknownType, it must be wrapped in an OptionType"
            )
        # Preserve option
        elif form.is_option and is_option(type_):
            child = determine_form_for_enforcing_type(form.content, type_.content, ctx)
            return ctx.choose(
                # If packed, pack content and build bytemask
                ak.forms.ByteMaskedForm(
                    "int8", child, True, parameters=type_._parameters
                ),
                # Otherwise, preserve existing option
                form.copy(content=child),
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
                ak.forms.IndexedForm(
                    getattr(form, "index", "i64"), child, parameters=form._parameters
                ),
            )
        # Add option
        elif not form.is_option and is_option(type_):
            return ak.forms.UnmaskedForm.simplified(
                determine_form_for_enforcing_type(form, type_.content, ctx)
            )
        # Indexed types
        elif form.is_indexed:
            child = determine_form_for_enforcing_type(form.content, type_, ctx)
            return ctx.choose(
                child.copy(
                    parameters=parameters_union(
                        form.content._parameters, form._parameters
                    )
                ),
                form.copy(content=child),
            )
        elif form.is_regular and is_regular(type_):
            # regular → regular requires same size!
            if form.size != type_.size:
                raise ValueError(
                    f"regular form has different size ({form.size}) to type ({type_.size})"
                )
            return determine_form_for_enforcing_type(form.content, type_.content, ctx)
        elif form.is_regular and is_list(type_):
            return ak.forms.ListOffsetForm(
                "i64",
                determine_form_for_enforcing_type(form.content, type_.content, ctx),
                parameters=type_.parameters,
            )
        elif form.is_list and is_list(type_):
            ...
        # Change dtype
        elif form.is_numpy and form.inner_shape == () and is_numpy(type_):
            return ak.forms.NumpyForm(type_.primitive)
        else:
            raise NotImplementedError

    numpy_layout = ak.to_layout([1.0, 2.0, 3.0])
    numpy_form = numpy_layout.form

    assert determine_form_for_enforcing_type(
        numpy_form, ak.types.from_datashape("int64", highlevel=False), Ctx()
    ) == (ak.forms.from_dict({"class": "NumpyArray", "primitive": "int64"}))

    assert determine_form_for_enforcing_type(
        numpy_form, ak.types.from_datashape("?int64", highlevel=False), Ctx()
    ) == ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64"))

    assert determine_form_for_enforcing_type(
        ak.forms.UnmaskedForm(numpy_form),
        ak.types.from_datashape("int64", highlevel=False),
        Ctx(),
    ) == ak.forms.NumpyForm("int64")
