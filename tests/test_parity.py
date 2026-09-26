from __future__ import annotations

import pytest
import pyfastx as reference
import numpy as np

import mojo_pyfastx as mpf
from mojo_pyfastx import _lib


def test_reverse_complement_matches_upstream_iupac():
    seq = "ACGTURYSWKMBDHVNacgturyswkmbdhvnXYZ"
    assert mpf.reverse_complement(seq) == reference.reverse_complement(seq)


def test_reverse_complement_simd_tail_and_multi_megabase_inputs_match_upstream():
    simd_tail = ("ACGTURYSWKMBDHVNacgturyswkmbdhvnXYZ" * 3) + "A"
    assert mpf.reverse_complement(simd_tail) == reference.reverse_complement(simd_tail)
    whole_vector = "ACGTN" * 3_355_443
    whole_vector_tail = ("ACGTN" * 3_355_443) + "AC"
    assert len(whole_vector) == 16 * 1024 * 1024 - 1
    assert len(whole_vector_tail) == 16 * 1024 * 1024 + 1
    assert mpf.reverse_complement(whole_vector) == reference.reverse_complement(whole_vector)
    assert mpf.reverse_complement(whole_vector_tail) == reference.reverse_complement(whole_vector_tail)


def test_histogram_simd_fast_path_scalar_fallback_and_tail():
    value = ("ACGTN" * 19) + "acgturyswkmbdhvnXYZ"
    expected = np.bincount(np.frombuffer(value.encode(), dtype=np.uint8), minlength=256)
    assert np.array_equal(_lib.histogram(value), expected)


def test_fasta_indexed_access_and_statistics_match_upstream(files):
    fasta, _ = files
    ours, theirs = mpf.Fasta(fasta), reference.Fasta(str(fasta))
    assert len(ours) == len(theirs)
    assert ours.keys() == list(theirs.keys())
    assert ours.size == theirs.size
    assert ours.composition == theirs.composition
    assert ours.gc_content == pytest.approx(theirs.gc_content)
    assert ours.gc_skew == pytest.approx(theirs.gc_skew)
    assert ours.type == theirs.type
    assert ours.mean == pytest.approx(theirs.mean)
    assert ours.median == theirs.median
    assert ours.nl(50) == theirs.nl(50)
    assert ours.count(5) == theirs.count(5)
    assert ours.longest.name == theirs.longest.name
    assert ours.shortest.name == theirs.shortest.name


def test_fasta_sequence_objects_match_upstream(files):
    fasta, _ = files
    ours, theirs = mpf.Fasta(fasta)["alpha"], reference.Fasta(str(fasta))["alpha"]
    assert str(ours) == str(theirs)
    assert ours.description == theirs.description
    assert ours.raw == theirs.raw
    assert ours.complement == theirs.complement
    assert ours.reverse == theirs.reverse
    assert ours.antisense == theirs.antisense
    assert ours.composition == theirs.composition
    assert ours.gc_content == pytest.approx(theirs.gc_content)
    assert ours.gc_skew == pytest.approx(theirs.gc_skew)
    assert ours[2] == theirs[2]
    assert ours[1:8].seq == theirs[1:8].seq
    assert ours.search("TNac") == theirs.search("TNac")
    assert list(ours) == list(theirs)


def test_fasta_fetch_flank_and_options_match_upstream(files):
    fasta, _ = files
    ours, theirs = mpf.Fasta(fasta), reference.Fasta(str(fasta))
    intervals = [(1, 3), (8, 12)]
    assert ours.fetch("alpha", (2, 8)) == theirs.fetch("alpha", (2, 8))
    assert ours.fetch("alpha", intervals) == theirs.fetch("alpha", intervals)
    assert ours.flank("alpha", 3, 8, 2) == theirs.flank("alpha", 3, 8, 2)
    assert list(mpf.Fasta(fasta, build_index=False, uppercase=True, full_name=True)) == list(
        reference.Fasta(str(fasta), build_index=False, uppercase=True, full_name=True)
    )


def test_fastq_indexed_access_and_statistics_match_upstream(files):
    _, fastq = files
    ours, theirs = mpf.Fastq(fastq), reference.Fastq(str(fastq))
    assert len(ours) == len(theirs)
    assert ours.keys() == list(theirs.keys())
    assert ours.size == theirs.size
    assert ours.composition == theirs.composition
    assert ours.gc_content == pytest.approx(theirs.gc_content)
    assert ours.encoding_type == theirs.encoding_type
    assert ours.phred == theirs.phred
    assert ours.minqual == theirs.minqual
    assert ours.maxqual == theirs.maxqual
    assert ours.avglen == pytest.approx(theirs.avglen)
    assert ours.minlen == theirs.minlen and ours.maxlen == theirs.maxlen


def test_read_objects_match_upstream(files):
    _, fastq = files
    ours, theirs = mpf.Fastq(fastq)["r1"], reference.Fastq(str(fastq))["r1"]
    assert str(ours) == str(theirs)
    assert ours.description == theirs.description
    assert ours.qual == theirs.qual
    assert ours.quali == theirs.quali
    assert ours.raw == theirs.raw
    assert ours.reverse == theirs.reverse
    assert ours.complement == theirs.complement
    assert ours.antisense == theirs.antisense


@pytest.mark.parametrize("low, high", [(33, 73), (59, 104), (64, 104), (66, 104)])
def test_fastq_quality_encoding_detection_matches_upstream(tmp_path, low, high):
    quality = "".join(chr(value) for value in range(low, high + 1))
    path = tmp_path / "quality.fq"
    path.write_text(f"@r\n{'A' * len(quality)}\n+\n{quality}\n")
    ours, theirs = mpf.Fastq(path), reference.Fastq(str(path))
    assert ours.phred == theirs.phred
    assert ours.encoding_type == theirs.encoding_type


def test_fastx_streaming_tuples_match_upstream_for_plain_and_gzip(files):
    fasta, fastq = files
    for path in (fasta, f"{fasta}.gz"):
        assert list(mpf.Fastx(path, comment=True, uppercase=True)) == list(
            reference.Fastx(str(path), comment=True, uppercase=True)
        )
    for path in (fastq, f"{fastq}.gz"):
        assert list(mpf.Fastx(path, comment=True)) == list(reference.Fastx(str(path), comment=True))


def test_indexed_fasta_and_fastq_support_gzip(files):
    fasta, fastq = files
    for path, cls in ((f"{fasta}.gz", mpf.Fasta), (f"{fastq}.gz", mpf.Fastq)):
        plain = cls(str(path)[:-3])
        compressed = cls(path)
        assert compressed.is_gzip
        assert compressed.keys() == plain.keys()
        assert compressed.composition == plain.composition


def test_ffi_preserves_all_byte_values_and_validates_arguments():
    raw = bytes(range(256))
    counts = _lib.histogram(raw)
    assert counts.dtype == np.dtype(np.int64)
    assert counts.flags.c_contiguous and counts.flags.writeable
    assert counts.tolist() == [1] * 256
    assert _lib.transform("", 2) == ""
    with pytest.raises(TypeError):
        _lib.transform(b"ACGT", 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _lib.transform("ACGT", 3)
    with pytest.raises(TypeError):
        _lib.histogram(bytearray(b"ACGT"))  # type: ignore[arg-type]


def test_ffi_null_and_invalid_calls_are_noops():
    # The exported C ABI must reject invalid pointers/modes before dereferencing.
    _lib.lib().mpf_transform(0, 0, 0, 1, 2)
    _lib.lib().mpf_transform(0, 0, 0, 0, 99)
    _lib.lib().mpf_count_bytes(0, 0, 1)


def test_index_and_validation_errors(files):
    fasta, fastq = files
    assert mpf.gzip_check(f"{fasta}.gz")
    assert not mpf.gzip_check(fasta)
    with pytest.raises(FileExistsError):
        mpf.Fasta(fasta.parent / "missing.fa")
    with pytest.raises(IndexError):
        mpf.Fasta(fasta)[99]
    with pytest.raises(KeyError):
        mpf.Fastq(fastq)["missing"]
    with pytest.raises(ValueError):
        mpf.Fasta(fasta).fetch("alpha", (8, 2))
    with pytest.raises(ValueError):
        mpf.Fasta(fasta).nl(101)
