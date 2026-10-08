"""Runtime contracts of the vendored doc parser, also checked by the core gate."""

from graph2mat.core._docstrings import FunctionDoc, ObjDoc, get_doc_object


def _documented():
    """A documented function.

    See Also
    --------
    helper : Related operation.
    """


def test_function_doc_preserves_default_and_accepts_parent_role():
    doc = FunctionDoc(_documented)
    assert ":func:`helper`" in str(doc)
    assert ":meth:`helper`" in doc.__str__(func_role="meth")


def test_doc_dispatch_preserves_noncallable_call_attribute():
    class NonCallable:
        __call__ = None

    assert isinstance(get_doc_object(_documented), FunctionDoc)
    assert isinstance(get_doc_object(NonCallable()), ObjDoc)
