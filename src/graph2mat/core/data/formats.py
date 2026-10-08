"""Module defining formats and conversion management.

Handling sparse matrices for associated 3D point clouds with basis functions is sometimes
not straightforward. For each different task (e.g. training a ML model, computing a property...)
there might be some data format that is more convenient. To the user (and the developer), converting
from any format to any other target format can be a pain. In `graph2mat`, we try to centralize
this task by:

- Having a class, `Formats`, that contains all the formats that we support.
- Having a class that manages the conversions between these formats: `ConversionManager``.

An instance of `ConversionManager` is available at ``graph2mat.conversions``.
"""
import inspect
from typing import Callable, Any, overload, Optional

from collections import ChainMap
import functools

import numpy as np
import scipy
import sisl

from .._docstrings import FunctionDoc


# --------------------------------------
#    REGISTER OF KNOWN FORMATS
# --------------------------------------
# We define all formats here, even if they are formats specific to bindings.
# This is to make it more clear what formats are available, otherwise it
# gets too messy with formats being defined all over the place.
class Formats:
    """Class holding all known formats.

    These are referenced by the conversion manager to understand what a function
    converts from and to.
    """

    #: Dictionary mapping aliases to formats.
    _aliases: dict[Any, str] = {}
    #: Dictionary mapping formats to aliases.
    _format_to_aliases: dict[str, list[Any]] = {}

    #: The format for a row block dictionary.
    BLOCK_DICT = "block_dict"
    #: The format for graph2mat's ``BasisMatrix``.
    BASISMATRIX = "basismatrix"

    #: The format for graph2mat's ``BasisConfiguration`` class.
    BASISCONFIGURATION = "basisconfiguration"
    #: The format for graph2mat's ``OrbitalConfiguration`` class.
    ORBITALCONFIGURATION = "orbitalconfiguration"

    #: Pseudoformat, arrays for edge and node values, without a container.
    NODESEDGES = "nodesedges"
    #: Pseudoformat, same as ``NODESEDGES`` but in torch tensors.
    TORCH_NODESEDGES = "torch_nodesedges"

    #: Numpy array
    NUMPY = "numpy"

    #: Scipy sparse COO matrix/array
    SCIPY_COO = "scipy_coo"
    #: Scipy sparse CSR matrix/array
    SCIPY_CSR = "scipy_csr"

    #: Sisl ``SparseOrbital`` class
    SISL = "sisl"
    #: Sisl ``Hamiltonian`` class
    SISL_H = "sisl_H"
    #: Sisl ``DensityMatrix`` class
    SISL_DM = "sisl_DM"
    #: Sisl ``EnergyDensityMatrix`` class
    SISL_EDM = "sisl_EDM"
    #: Sisl ``Geometry`` class, doesn't contain matrix information.
    SISL_GEOMETRY = "sisl_geometry"
    #: Pseudoformat, path to a file from which sisl can read a matrix's data.
    SISL_SILE = "sisl_sile"

    #: The format for graph2mat's ``BasisMatrixData`` class
    BASISMATRIXDATA = "basismatrixdata"
    #: The format for graph2mat's ``TorchBasisMatrixData`` class.
    TORCH_BASISMATRIXDATA = "torch_basismatrixdata"

    #: Torch tensor
    TORCH = "torch"
    #: Torch sparse COO tensor
    TORCH_COO = "torch_coo"
    #: Torch sparse CSR tensor
    TORCH_CSR = "torch_csr"

    #: List of all pseudoformats. This is useful for the ConversionManager for
    #: example, as conversion functions don't have a single argument to pass the
    #: data to convert. A function to convert from a pseudo format accepts the
    #: data as multiple arguments.
    _pseudo_formats: list[str] = [NODESEDGES, TORCH_NODESEDGES]

    @classmethod
    def add_alias(cls, fmt: str, *aliases: Any):
        """Add an alias for a format.

        Parameters
        ----------
        fmt :
            The format name.
        aliases :
            The aliases that will be associated with the format.
            They don't need to be strings, they can be e.g. a
            class.
        """
        for alias in aliases:
            cls._aliases[alias] = fmt

        if fmt not in cls._format_to_aliases:
            cls._format_to_aliases[fmt] = []

        cls._format_to_aliases[fmt].extend(aliases)

    @classmethod
    def string_to_attr_name(cls, format_string: str) -> str:
        """Get the attribute name that corresponds to a given format string.

        This function is quite slow.

        Parameters
        ----------
        format_string :
            The format string.

        Returns
        -------
        attr_name :
            The attribute name.
        """
        for attr in dir(cls):
            if getattr(cls, attr) == format_string:
                return attr
        else:
            raise ValueError(f"Format '{format_string}' not found")


# --------------------------------------
#    MAPPING PYTHON TYPES TO FORMATS
# --------------------------------------
# Numpy types
Formats.add_alias(Formats.NUMPY, np.ndarray)
# Scipy types
Formats.add_alias(Formats.SCIPY_COO, scipy.sparse.coo_matrix)
Formats.add_alias(Formats.SCIPY_COO, scipy.sparse.coo_array)
Formats.add_alias(Formats.SCIPY_CSR, scipy.sparse.csr_matrix)
Formats.add_alias(Formats.SCIPY_CSR, scipy.sparse.csr_array)
# Sisl types
Formats.add_alias(Formats.SISL, sisl.SparseOrbital)
Formats.add_alias(Formats.SISL_H, sisl.Hamiltonian)
Formats.add_alias(Formats.SISL_DM, sisl.DensityMatrix)
Formats.add_alias(Formats.SISL_EDM, sisl.EnergyDensityMatrix)
Formats.add_alias(Formats.SISL_GEOMETRY, sisl.Geometry)

# --------------------------------------
#      MANAGEMENT OF CONVERSIONS
# --------------------------------------


class ConversionManager:
    """Manages the conversions between formats.

    This class centralizes the handling of conversions between formats.
    It uses the formats defined in the ``Formats`` class.

    Examples
    --------
    The conversion manager needs to be instantiated in order to be used:

    .. code-block:: python

        conversions = ConversionManager()

    Notice that by doing that, you get an empty conversion manager (with no
    implemented conversions). ``graph2mat`` already **provides an instantiated
    conversion manager with all the implemented conversions**. It can be imported
    like:

    .. code-block:: python

        from graph2mat import conversions

    Then, converters can be registered using the ``register_converter`` method:

    .. code-block:: python

        def my_converter(data: np.ndarray) -> scipy.sparse.coo_matrix:
            ...

        conversions.register_converter(Formats.NUMPY, Formats.SCIPY_COO, my_converter)

    Or using the ``converter`` decorator:

    .. code-block:: python

        @conversions.converter(Formats.NUMPY, Formats.SCIPY_COO)
        def my_converter(data: np.ndarray) -> scipy.sparse.coo_matrix:
            ...

    The registered converters can be retrieved using the ``get_converter`` method:

    .. code-block:: python

        converter = conversions.get_converter(Formats.NUMPY, Formats.SCIPY_COO)

        # Matrix as a numpy array
        array = np.random.rand(10, 10)
        # Use the converter to get the matrix as a scipy sparse COO matrix
        sparse_coo = converter(array)

    They converters are also registered as attributes of the conversion manager,
    with the names being `<source>_to_<target>`. For example, to get the numpy
    to scipy COO converter, one can also do:

    .. code-block:: python

        converter = conversions.numpy_to_scipy_coo

    .. note::

        When writing code that should be future-proof (e.g. inside a package), we
        recommend using ``get_converter`` with the formats retreived from ``Formats``.
        In the very unlikely event that some format name changes, the error raised
        will be more informative.

    See Also
    --------
    Formats
        The class that defines all formats.
    """

    #: Dictionary holding all explicitly registered converters.
    #: Keys are tuples (source, target) and values are the converter functions.
    _registered_converters: dict[tuple[str, str], Callable] = {}

    #: Dictionary holding all automatically generated converters.
    #: Keys are tuples (source, target) and values are the converter functions.
    _autodef_converters: dict[tuple[str, str], Callable] = {}

    #: ChainMap of all converters, with the registered converters taking precedence.
    _converters: ChainMap[tuple[str, str], Callable]

    #: List of callbacks that are called every time a new converter is registered.
    _callbacks: list[Callable[[str, str, Callable, "ConversionManager"], Any]] = []

    def __init__(self):
        self._registered_converters = {}
        self._autodef_converters = {}
        self._converters = ChainMap(
            self._registered_converters, self._autodef_converters
        )
        self._callbacks = []

    def get_converter(self, source: str, target: str) -> Callable:
        """Get a converter function between two formats.

        It raises a ``KeyError`` if no converter is found.

        Parameters
        ----------
        source :
            The source format.
        target :
            The target format.

        Returns
        -------
        converter :
            The converter function for the given formats.
        """
        try:
            return self._converters[(source, target)]
        except KeyError:
            raise KeyError(f"No converter found from '{source}' to '{target}'")

    def has_converter(self, source: str, target: str) -> bool:
        """Check if a converter exists between two formats.

        Parameters
        ----------
        source :
            The source format.
        target :
            The target format.
        """
        return (source, target) in self._converters

    def register_converter(
        self,
        source: str,
        target: str,
        converter: Callable,
        exists_ok: bool = False,
        autodef: bool = False,
    ):
        """Register a converter function between two formats.

        Parameters
        ----------
        source :
            The source format.
        target :
            The target format.
        converter :
            The function that converts from source to target.
        exists_ok :
            If ``False``, raises a ``KeyError`` error if a converter
            from ``source`` to ``target`` already exists.
        autodef :
            Whether this is an automatically generated converter.
            Only set to ``True`` by internal calls, should not be set by the user.
        """

        if not exists_ok and (source, target) in self._registered_converters:
            raise KeyError(f"Converter from '{source}' to '{target}' already exists")

        if autodef:
            self._autodef_converters[(source, target)] = converter
        else:
            # Add the converter
            self._converters[(source, target)] = converter

        setattr(self, f"{source}_to_{target}", converter)

        if autodef:
            return

        # If the converter only has one argument, then we can try to extend
        # the list of converters by chaining them.
        # E.g. the registered converter is from A to B with a single argument,
        # and there is a registered converter from B to C. Then we can automatically
        # register a converter from A to C that does A -> B -> C, because it doesn't
        # require any additional information.
        try:
            sig = inspect.signature(converter)
            if len(sig.parameters) == 1:

                def _auto_converter_callback(
                    new_source, new_target, new_converter, manager
                ):
                    if new_target == source:
                        # Trigger an expansion to the right:
                        # source -> target [-> new_target]
                        if new_source == target or manager.has_converter(
                            new_source, target
                        ):
                            return

                        manager.register_expanded_converter(
                            new_source, new_target, source, target, converter
                        )

                    if target == new_source:
                        # Trigger and expansion to the left:
                        # [new_source ->] source -> target
                        if source == new_target or manager.has_converter(
                            source, new_target
                        ):
                            return

                        if new_source in Formats._pseudo_formats:
                            return

                        manager.register_expanded_converter(
                            new_source, new_target, source, target, converter
                        )

                self.add_callback(_auto_converter_callback, retroactive=True)
        except:
            pass

        for callback in self._callbacks:
            callback(source, target, converter, self)

    def register_expanded_converter(
        self,
        old_source: str,
        old_target: str,
        expansion_source: str,
        expansion_target: str,
        expansion: Callable,
    ):
        """Registers a converter that is an expansion of an existing one.

        This function takes care of modifying the signature, docstring etc... so that
        the user still sees some helpful information when inspecting the converter
        e.g. in a Jupyter notebook or in the documentation.

        The function will automatically detect whether the original converter
        is to be expanded to the left or to the right, and will create a new
        converter that chains the original one with the expansion.

        Parameters
        ----------
        old_source
            The source format of the original converter.
        old_target
            The target format of the original converter.
        expansion_source
            The source format of the expansion.
        expansion_target
            The target format of the expansion.
        expansion
            The function that expands the original converter.
        """
        # Information about the original converter
        old_converter = self.get_converter(old_source, old_target)
        old_signature = inspect.signature(old_converter)
        old_annotations = getattr(old_converter, "__annotations__", {})
        old_fdoc = FunctionDoc(old_converter)
        old_path = getattr(old_converter, "_converter_path", (old_source, old_target))
        explicit_converter = getattr(
            old_converter, "_explicit_converter", (old_source, old_target)
        )
        # Information about the expansion
        expansion_signature = inspect.signature(expansion)
        expansion_fdoc = FunctionDoc(expansion)
        expansion_annotations = getattr(expansion, "__annotations__", {})

        if expansion_target == old_source:
            # Converter expanded "to the left"
            path = (expansion_source, *old_path)

            # The first argument is getting replaced to accept the input of
            # the expansion instead. Here we do all the work to erase any
            # trace of the original first argument, and document the new
            # first argument instead. The name of the new first argument
            # might already be taken by another parameter, in which case
            # we rename the original first argument to "first_arg".
            first_param = list(expansion_signature.parameters.values())[0]
            orig_first_param_name = first_param.name
            old_params = list(old_signature.parameters.values())

            if first_param.name in old_signature.parameters:
                first_param = first_param.replace(name="first_arg")

            signature = old_signature.replace(
                parameters=[first_param, *old_params[1:]],
            )

            new_params_doc = list(
                filter(
                    lambda p: p.name.replace(":", "") != old_params[0].name,
                    old_fdoc["Parameters"],
                )
            )

            for p in expansion_fdoc["Parameters"]:
                if p.name.replace(":", "").strip() == orig_first_param_name:
                    p = p.__class__(name=first_param.name, type=p.type, desc=p.desc)
                    new_params_doc.insert(0, p)
                    break

            old_fdoc["Parameters"] = new_params_doc
            annotations = {
                **old_annotations,
                first_param.name: expansion_annotations.get(orig_first_param_name),
            }

            # Define the chained converter
            def _composite(*args, **kwargs):
                # Find the first argument, which will be passed to the expansion
                if len(args) > 0:
                    first_arg, *remaining_args = args
                else:
                    first_arg = kwargs.pop(first_param.name)
                    remaining_args = list(args)
                # Call the converter with the first argument converted
                return old_converter(expansion(first_arg), *remaining_args, **kwargs)

        elif expansion_source == old_target:
            # Converter expanded "to the right"
            path = (*old_path, expansion_target)

            def _composite(*args, **kwargs):
                return expansion(old_converter(*args, **kwargs))

            # This case is simpler, we just need to modify the return information
            signature = old_signature.replace(
                return_annotation=expansion_signature.return_annotation
            )
            old_fdoc["Returns"] = expansion_fdoc["Returns"]
            annotations = {**old_annotations, "return": expansion_annotations["return"]}
        else:
            raise ValueError(
                f"Can't expand converter '{old_source}'->'{old_target}' with expansion '{expansion_source}' -> '{expansion_target}'"
            )

        # Make the function look nice to the outside world
        _composite.__name__ = f"{path[0]}_to_{path[-1]}"
        _composite.__qualname__ = f"{path[0]}_to_{path[-1]}"
        setattr(_composite, "__signature__", signature)
        _composite.__annotations__ = annotations
        setattr(_composite, "_converter_path", path)
        setattr(_composite, "_explicit_converter", explicit_converter)

        prev_format = path[0]
        full_path_string = ""
        for fmt in path[1:]:
            if full_path_string.endswith("] -> "):
                continue

            if prev_format == explicit_converter[0] and fmt == explicit_converter[-1]:
                full_path_string += f" [ ``{prev_format}`` -> ``{fmt}`` ] -> "
            else:
                full_path_string += f" ``{prev_format}`` -> "

            prev_format = fmt
        if full_path_string.endswith(f"``{fmt}`` ] -> "):
            full_path_string = full_path_string.removesuffix(" -> ")
        else:
            full_path_string += f"``{fmt}``"

        old_fdoc["Summary"] = [f"Conversion from '{path[0]}' to '{path[-1]}'."]
        old_fdoc["Extended Summary"] = [
            "\n\nThis function has been defined automatically."
            "\n\nThe full path of the conversion is (original converter in square brackets):\n\n"
            + full_path_string,
        ]

        old_fdoc["See Also"] = [
            (
                [(f"{explicit_converter[0]}_to_{explicit_converter[-1]}", None)],
                ["The original converter that has been expanded to create this one."],
            ),
        ]

        for k in old_fdoc:
            if k not in [
                "Summary",
                "Extended Summary",
                "Parameters",
                "Returns",
                "See Also",
            ]:
                old_fdoc[k] = old_fdoc[k].__class__()

        docstring = str(old_fdoc)
        _composite.__doc__ = docstring[docstring.find("\n") :].lstrip()

        # And register it
        self.register_converter(path[0], path[-1], _composite, autodef=True)

    @overload
    def converter(
        self, source: str, target: str, /, *, exists_ok: bool = False
    ) -> Callable[[Callable], Callable]:
        ...

    @overload
    def converter(self, converter: Callable, /, *, exists_ok: bool = False) -> Callable:
        ...

    def converter(self, *args, exists_ok: bool = False):
        """Decorator to register a converter while defining a function.

        Examples
        --------
        There are two ways to use this decorator:

        1. As a decorator with two arguments:

        .. code-block:: python

            @converter("source_format", "target_format")
            def my_converter(...):
                ...

        Where ``source_format`` and ``target_format`` are strings representing
        the input and output formats of the converter.

        2. As a no argument decorator:

        .. code-block:: python

            @converter
            def my_converter(data: np.ndarray) -> scipy.sparse.coo_matrix:
                ...

        In which case the source and target formats are inferred
        from the function signature.

        """
        if len(args) == 1:
            # This is the case when the decorator is used without arguments.
            # Then the first argument is the function to be decorated.
            converter = args[0]

            # Find out source and target from the function signature
            # if the function only has one argument
            sig = inspect.signature(converter)
            if len(sig.parameters) != 1:
                raise ValueError(
                    "Can't guess source for converter with more than one argument, please provide them explicitly"
                )
            _, param = next(iter(sig.parameters.items()))
            source = param.annotation
            target = sig.return_annotation

            # Try to get the format that corresponds to each of the types.
            try:
                source = Formats._aliases[source]
            except KeyError:
                raise ValueError(
                    f"Could not guess format from argument annotation '{source}'"
                )

            try:
                target = Formats._aliases[target]
            except KeyError:
                raise ValueError(
                    f"Could not guess format from return annotation '{target}'"
                )

            self.register_converter(source, target, converter, exists_ok=exists_ok)
            return converter
        else:
            # This is the case when the decorator is used with two arguments.
            source, target = args

            def wrapper(converter):
                self.register_converter(source, target, converter, exists_ok=exists_ok)
                return converter

        return wrapper

    def add_callback(
        self,
        callback: Callable[[str, str, Callable, "ConversionManager"], Any],
        retroactive: bool = False,
    ):
        """Add a function that will be called every time a new converter is registered.

        Parameters
        ----------
        callback :
            The callback function. It will receive the source format, target format,
            the converter function being registered and the ConversionManager instance.
        retroactive : bool
            If True, the callback will be called for all converters that have been
            previously registered.
        """
        self._callbacks.append(callback)
        if retroactive:
            items = list(self._converters.items())
            for (source, target), converter in items:
                callback(source, target, converter, self)

    def _repr_html_(self) -> str:
        table = "<table><tbody>"
        table += f"<tr><th>Source</th><th>Target</th></tr>"

        for source, target in sorted(self._converters):
            table += f"<tr><td>{source}</td><td>{target}</td></tr>"

        table += "</tbody></table>"

        return table

    def get_available_targets(self, source: str) -> list[str]:
        """For a given format, return all formats it can be converted to.

        Parameters
        ----------
        source :
            The source format.
        """
        return [target for (s, target) in self._converters if s == source]

    def get_available_sources(self, target: str) -> list[str]:
        """For a given format, return all formats it can be converted from.

        Parameters
        ----------
        target :
            The target format
        """
        return [source for (source, t) in self._converters if t == target]


conversions = ConversionManager()
