# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import collections
import functools
import inspect
from collections.abc import Iterable
from itertools import chain

import numpy
from packaging.version import parse as parse_version

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import backend_of, common_backend
from awkward._behavior import (
    behavior_of,
    find_custom_cast,
    find_ufunc,
    find_ufunc_generic,
)
from awkward._categorical import as_hashable
from awkward._layout import wrap_layout
from awkward._nplikes import to_nplike
from awkward._parameters import parameters_intersect
from awkward._regularize import is_non_string_like_iterable
from awkward._typing import Any, Iterator, Mapping
from awkward._util import Sentinel
from awkward.contents.numpyarray import NumpyArray
from awkward.units import get_unit_registry

# NumPy 1.13.1 introduced NEP13, without which Awkward ufuncs won't work, which
# would be worse than lacking a feature: it would cause unexpected output.
# NumPy 1.17.0 introduced NEP18, which is optional (use ak.* instead of np.*).
if parse_version(numpy.__version__) < parse_version("1.13.1"):
    raise ImportError("NumPy 1.13.1 or later required")


UNSUPPORTED = Sentinel("UNSUPPORTED", __name__)


def convert_to_array(layout, args, kwargs):
    out = ak.operations.to_numpy(layout, allow_missing=False)
    if args == () and kwargs == {}:
        return out
    else:
        return numpy.array(out, *args, **kwargs)


implemented = {}


def _find_backends(args: Iterable) -> Iterator[Backend]:
    """
    Args:
        args: iterable of objects to visit

    Yields the encountered backends of layout / array-like arguments encountered
    in the argument list.
    """
    stack = collections.deque(args)
    while stack:
        arg = stack.popleft()
        # If the argument declares a backend, easy!
        backend = backend_of(arg, default=None)
        if backend is not None:
            yield backend
        # Otherwise, traverse into supported sequence types
        elif isinstance(arg, (tuple, list)):
            stack.extend(arg)


def _to_rectilinear(arg, backend: Backend):
    # Is this object something we already associate with a backend?
    arg_backend = backend_of(arg, default=None)
    if arg_backend is not None:
        arg_nplike = arg_backend.nplike
        # Is this argument already in a backend-supported form?
        if arg_nplike.is_own_array(arg):
            # Convert to the appropriate nplike
            return to_nplike(
                arg_nplike.asarray(arg), backend.nplike, from_nplike=arg_nplike
            )
        # Otherwise, cast to layout and convert
        else:
            layout = ak.to_layout(arg, allow_record=False, allow_other=False)
            return layout.to_backend(backend).to_backend_array(allow_missing=True)
    elif isinstance(arg, tuple):
        return tuple(_to_rectilinear(x, backend) for x in arg)
    elif isinstance(arg, list):
        return [_to_rectilinear(x, backend) for x in arg]
    elif is_non_string_like_iterable(arg):
        raise TypeError(
            f"encountered an unsupported iterable value {arg!r} whilst converting arguments to NumPy-friendly "
            f"types. If this argument should be supported, please file a bug report."
        )
    else:
        return arg


def array_function(func, types, args, kwargs: dict[str, Any], behavior: Mapping | None):
    function = implemented.get(func)
    if function is not None:
        return function(*args, **kwargs)
    # Use NumPy's implementation
    else:
        backend = common_backend(_find_backends(chain(args, kwargs.values())))

        rectilinear_args = tuple(_to_rectilinear(x, backend) for x in args)
        rectilinear_kwargs = {k: _to_rectilinear(v, backend) for k, v in kwargs.items()}
        result = func(*rectilinear_args, **rectilinear_kwargs)
        # We want the result to be a layout (this will fail for functions returning non-array convertibles)
        out = ak.operations.ak_to_layout._impl(
            result, allow_record=True, allow_other=True, regulararray=True
        )
        return wrap_layout(out, behavior=behavior, allow_other=True)


def implements(numpy_function):
    def decorator(function):
        signature = inspect.signature(function)
        unsupported_names = {
            p.name for p in signature.parameters.values() if p.default is UNSUPPORTED
        }

        @functools.wraps(function)
        def ensure_valid_args(*args, **kwargs):
            parameters = signature.bind(*args, **kwargs)
            provided_invalid_names = parameters.arguments.keys() & unsupported_names
            if provided_invalid_names:
                names = ", ".join(provided_invalid_names)
                raise TypeError(
                    f"Awkward NEP-18 overload was provided with unsupported argument(s): {names}"
                )
            return function(*args, **kwargs)

        implemented[getattr(numpy, numpy_function)] = ensure_valid_args
        return function

    return decorator


def _array_ufunc_custom_cast(inputs, behavior: Mapping | None, backend):
    args = [
        wrap_layout(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]

    nextinputs = []
    for x in args:
        cast_fcn = find_custom_cast(x, behavior)
        if cast_fcn is not None:
            x = cast_fcn(x)
        # String conversion
        elif isinstance(x, (str, bytes)):
            x = ak.to_layout([x])
        maybe_layout = ak.operations.to_layout(x, allow_record=True, allow_other=True)
        if isinstance(maybe_layout, (ak.contents.Content, ak.record.Record)):
            maybe_layout = maybe_layout.to_backend(backend)

        nextinputs.append(maybe_layout)
    return nextinputs


def _array_ufunc_adjust(
    custom, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
    args = [
        wrap_layout(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]
    out = custom(*args, **kwargs)
    if not isinstance(out, tuple):
        out = (out,)

    return tuple(
        x.layout if isinstance(x, (ak.highlevel.Array, ak.highlevel.Record)) else x
        for x in out
    )


def _array_ufunc_adjust_apply(
    apply_ufunc, ufunc, method, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
    nextinputs = [wrap_layout(x, behavior, allow_other=True) for x in inputs]
    out = apply_ufunc(ufunc, method, nextinputs, kwargs)

    if out is NotImplemented:
        return None
    else:
        if not isinstance(out, tuple):
            out = (out,)
        return tuple(
            x.layout if isinstance(x, (ak.highlevel.Array, ak.highlevel.Record)) else x
            for x in out
        )


def _array_ufunc_signature(ufunc, inputs):
    signature = [ufunc]
    for x in inputs:
        if isinstance(x, ak.contents.Content):
            record_name, list_name = x.parameter("__record__"), x.parameter("__list__")
            if record_name is not None:
                signature.append(record_name)
            elif list_name is not None:
                signature.append(list_name)
            elif isinstance(x, NumpyArray):
                signature.append(x.dtype.type)
            else:
                signature.append(None)
        else:
            signature.append(type(x))

    return tuple(signature)


def _array_ufunc_categorical(
    ufunc, method: str, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
    if (
        ufunc is numpy.equal
        and len(inputs) == 2
        and isinstance(inputs[0], ak.contents.Content)
        and inputs[0].is_indexed
        and inputs[0].parameter("__array__") == "categorical"
        and isinstance(inputs[1], ak.contents.Content)
        and inputs[1].is_indexed
        and inputs[1].parameter("__array__") == "categorical"
    ):
        assert method == "__call__"
        one, two = inputs

        one_index = numpy.asarray(one.index)
        two_index = numpy.asarray(two.index)
        one_content = wrap_layout(one.content, behavior)
        two_content = wrap_layout(two.content, behavior)

        if len(one_content) == len(two_content) and ak.operations.all(
            one_content == two_content, axis=None
        ):
            one_mapped = one_index

        else:
            one_list = ak.operations.to_list(one_content)
            two_list = ak.operations.to_list(two_content)
            one_hashable = [as_hashable(x) for x in one_list]
            two_hashable = [as_hashable(x) for x in two_list]
            two_lookup = {x: i for i, x in enumerate(two_hashable)}

            one_to_two = numpy.empty(len(one_hashable) + 1, dtype=numpy.int64)
            for i, x in enumerate(one_hashable):
                one_to_two[i] = two_lookup.get(x, len(two_hashable))
            one_to_two[-1] = -1

            one_mapped = one_to_two[one_index]

        out = one_mapped == two_index
        return (ak.contents.NumpyArray(out),)
    else:
        nextinputs = []
        for x in inputs:
            if isinstance(x, ak.contents.Content) and x.is_indexed:
                nextinputs.append(wrap_layout(x.project(), behavior=behavior))
            else:
                nextinputs.append(wrap_layout(x, behavior=behavior, allow_other=True))

        out = getattr(ufunc, method)(*nextinputs, **kwargs)
        if not isinstance(out, tuple):
            out = (out,)
        return tuple(ak.to_layout(x, allow_other=True) for x in out)


def _array_ufunc_custom_units(ufunc, inputs, kwargs, behavior):
    registry = get_unit_registry()
    if registry is None:
        return None

    # Check if we have units
    for x in inputs:
        if isinstance(x, ak.contents.Content) and x.parameter("__units__"):
            break
        elif isinstance(x, registry.Quantity):
            break
    # Exit now, if not!
    else:
        return None

    # Wrap all Awkward Arrays with
    nextinputs = []
    for x in inputs:
        if isinstance(x, ak.contents.Content):
            nextinputs.append(
                registry.Quantity(
                    ak.with_parameter(x, "__units__", None, behavior=behavior),
                    x.parameter("__units__"),
                )
            )
        else:
            nextinputs.append(x)
    out = ufunc(*nextinputs, **kwargs)
    if not isinstance(out, tuple):
        out = (out,)

    nextout = []
    for qty in out:
        assert isinstance(qty, registry.Quantity)
        assert isinstance(qty.magnitude, ak.Array)
        nextout.append(
            ak.with_parameter(
                qty.magnitude, "__units__", str(qty.units), highlevel=False
            )
        )
    return tuple(nextout)


def _array_ufunc_string_likes(
    ufunc, method: str, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
    assert method == "__call__"

    if (
        ufunc in (numpy.equal, numpy.not_equal)
        and len(inputs) == 2
        and isinstance(inputs[0], ak.contents.Content)
        and isinstance(inputs[1], ak.contents.Content)
        and inputs[0].parameter("__array__") in ("string", "bytestring")
        and inputs[1].parameter("__array__") == inputs[0].parameter("__array__")
    ):
        left, right = inputs
        nplike = left.backend.nplike

        left = ak.without_parameters(left, highlevel=False)
        right = ak.without_parameters(right, highlevel=False)

        # first condition: string lengths must be the same
        counts1 = nplike.asarray(
            ak._do.reduce(left, ak._reducers.Count(), axis=-1, mask=False)
        )
        counts2 = nplike.asarray(
            ak._do.reduce(right, ak._reducers.Count(), axis=-1, mask=False)
        )

        out = counts1 == counts2

        # only compare characters in strings that are possibly equal (same length)
        possible = nplike.logical_and(out, counts1)
        possible_counts = counts1[possible]

        if len(possible_counts) > 0:
            onepossible = left[possible]
            twopossible = right[possible]
            reduced = ak.operations.all(
                wrap_layout(onepossible) == wrap_layout(twopossible),
                axis=-1,
                highlevel=False,
            )
            # update same-length strings with a verdict about their characters
            out[possible] = reduced.data

        if ufunc is numpy.not_equal:
            out = nplike.logical_not(out)

        return (ak.contents.NumpyArray(out),)


def array_ufunc(ufunc, method: str, inputs, kwargs: dict[str, Any]):
    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    behavior = behavior_of(*inputs)
    backend = backend_of(*inputs)

    inputs = _array_ufunc_custom_cast(inputs, behavior, backend)

    def action(inputs, **ignore):
        # Do we have any units in the mix? If so, delegate to `pint` to perform
        # the ufunc dispatch. This will re-enter `array_ufunc`, but without units
        # NOTE: there's nothing preventing us from handling units for non-NumpyArray
        #       contents, but for now we restrict ourselves to NumpyArray (in the
        #       NumpyArray constructor). By running _before_ the custom machinery,
        #       custom user ufuncs can avoid needing to worry about units
        out = _array_ufunc_custom_units(ufunc, inputs, kwargs, behavior)
        if out is not None:
            return out

        signature = _array_ufunc_signature(ufunc, inputs)
        # Do we have a custom (specific) ufunc (an override of the given ufunc)?
        custom = find_ufunc(behavior, signature)
        if custom is not None:
            return _array_ufunc_adjust(custom, inputs, kwargs, behavior)

        # Do we have a custom generic ufunc override (a function that accepts _all_ ufuncs)?
        contents = [x for x in inputs if isinstance(x, ak.contents.Content)]
        for x in contents:
            apply_ufunc = find_ufunc_generic(ufunc, x, behavior)
            if apply_ufunc is not None:
                out = _array_ufunc_adjust_apply(
                    apply_ufunc, ufunc, method, inputs, kwargs, behavior
                )
                if out is not None:
                    return out

        # Do we have any categoricals?
        if any(
            x.is_indexed and x.parameter("__array__") == "categorical" for x in contents
        ):
            out = _array_ufunc_categorical(ufunc, method, inputs, kwargs, behavior)
            if out is not None:
                return out

        # Do we have any strings?
        if any(
            x.is_list and x.parameter("__array__") in ("string", "bytestring")
            for x in contents
        ):
            out = _array_ufunc_string_likes(ufunc, method, inputs, kwargs, behavior)
            if out is not None:
                return out

            # Do we have all-strings? If so, we can't proceed
            if all(
                x.is_list and x.parameter("__array__") in ("string", "bytestring")
                for x in contents
            ):
                raise TypeError(
                    "{}.{} is not implemented for string types. "
                    "To register an implementation, add a name to these string(s) and register a behavior overload".format(
                        type(ufunc).__module__, ufunc.__name__
                    )
                )

        if ufunc is numpy.matmul:
            raise NotImplementedError(
                "matrix multiplication (`@` or `np.matmul`) is not yet implemented for Awkward Arrays"
            )

        if all(
            isinstance(x, NumpyArray) or not isinstance(x, ak.contents.Content)
            for x in inputs
        ):
            nplike = backend.nplike

            # Broadcast parameters against one another
            parameters = functools.reduce(
                parameters_intersect, [x._parameters for x in contents]
            )

            args = []
            for x in inputs:
                if isinstance(x, NumpyArray):
                    args.append(x._raw(nplike))
                else:
                    args.append(x)

            # Give backend a chance to change the ufunc implementation
            impl = backend.prepare_ufunc(ufunc)

            # Invoke ufunc
            result = impl(*args, **kwargs)

            return (NumpyArray(result, backend=backend, parameters=parameters),)

        # Do we have exclusively nominal types without custom overloads?
        if all(
            x.parameter("__list__") is not None or x.parameter("__record__") is not None
            for x in contents
        ):
            error_message = []
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    if x.parameter("__list__") is not None:
                        error_message.append(x.parameter("__list__"))
                    elif x.parameter("__record__") is not None:
                        error_message.append(x.parameter("__record__"))
                    else:
                        error_message.append(type(x).__name__)
                else:
                    error_message.append(type(x).__name__)
            raise TypeError(
                "no {}.{} overloads for custom types: {}".format(
                    type(ufunc).__module__, ufunc.__name__, ", ".join(error_message)
                )
            )

        return None

    if sum(int(isinstance(x, ak.contents.Content)) for x in inputs) == 1:
        where = None
        for i, x in enumerate(inputs):
            if isinstance(x, ak.contents.Content):
                where = i
                break
        assert where is not None

        nextinputs = list(inputs)

        def unary_action(layout, **ignore):
            nextinputs[where] = layout
            result = action(tuple(nextinputs), **ignore)
            if result is None:
                return None
            else:
                assert isinstance(result, tuple) and len(result) == 1
                return result[0]

        out = ak._do.recursively_apply(
            inputs[where],
            unary_action,
            behavior,
            function_name=ufunc.__name__,
            allow_records=False,
        )

    else:
        out = ak._broadcasting.broadcast_and_apply(
            inputs, action, behavior, allow_records=False, function_name=ufunc.__name__
        )
        assert isinstance(out, tuple) and len(out) == 1
        out = out[0]

    return wrap_layout(out, behavior)


def action_for_matmul(inputs):
    raise NotImplementedError
