"""Read-only audit boundaries and SCF parsing, without external datasets."""

import hashlib

import pytest


def test_source_guard_rejects_escaping_symlink(tmp_path):
    from auditorias_mejoras.audit_multitarget_w90 import checked_path

    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("do not read")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        checked_path(root, root / "escape")
    inside = root / "data"
    inside.write_text("allowed")
    assert checked_path(root, inside) == inside


def test_hash_guard_detects_changed_bytes(tmp_path):
    from auditorias_mejoras.audit_multitarget_w90 import verified_file

    source = tmp_path / "source"
    source.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    assert verified_file(tmp_path, source, expected) == expected
    source.write_bytes(b"abd")
    with pytest.raises(ValueError, match="hash"):
        verified_file(tmp_path, source, expected)
    assert source.read_bytes() == b"abd"


def test_scf_parser_does_not_count_unconverged_frame_as_success():
    from auditorias_mejoras.audit_multitarget_w90 import scf_summary

    text = ("Begin MD step = 1\nSCF cycle converged after 5 iterations\n"
            "Begin MD step = 2\nSCF cycle did not converge\nJob completed\n")
    result = scf_summary(text)
    assert result["md_steps"] == 2 and result["converged_cycles"] == 1
    assert result["all_steps_converged"] is False
    assert scf_summary("SCF cycle converged after 3 iterations\n")["all_steps_converged"] is False


def test_native_tshs_placeholder_species_is_not_a_geometry_failure():
    import sisl
    from auditorias_mejoras.audit_multitarget_w90 import validate_native_geometry

    native = sisl.Geometry([[0., 0., 0.]], atoms=sisl.Atom("H", orbitals=[sisl.Orbital(2.)] * 4), lattice=[10.] * 3)
    attached = sisl.Geometry([[0., 0., 0.]], atoms=sisl.Atom("C", orbitals=[sisl.Orbital(2.)] * 4), lattice=[10.] * 3)
    assert validate_native_geometry(native, attached) == {"xyz_max": 0., "cell_max": 0.}
    attached.xyz[0, 0] = .01
    with pytest.raises(ValueError, match="geometry"):
        validate_native_geometry(native, attached)
