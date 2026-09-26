# mojo-pyfastx

`mojo-pyfastx` is a focused Mojo port of [pyfastx](https://pypi.org/project/pyfastx/)
for FASTA/FASTQ indexed access. It provides a pyfastx-shaped Python API while moving
the byte-heavy sequence kernels—reverse, complement, reverse-complement, and byte
histograms used for composition and quality summaries—into Mojo.

The Python package is named `mojo_pyfastx`; use `import mojo_pyfastx as pyfastx` in code
that expects the covered pyfastx API.

## Covered subset

- `Fasta`, `Fastq`, and `Fastx` for plain-text and gzip-compressed input.
- Name and numeric indexed access, iteration, `keys`, FASTA `fetch`/`flank`, length
  statistics, composition, GC content/skew, and FASTQ Phred summaries.
- `Sequence` and `Read` properties: `seq`, `raw`, `description`, and reverse/complement/
  antisense; `Sequence` also supports slices.
- `reverse_complement()` and `gzip_check()`.

It does not implement pyfastx's persistent SQLite `.fxi` cache, `FastaKeys`/`FastqKeys`,
or command-line splitting tools. Its index is in memory, so opening a file parses it each
time and retains the sequence and quality strings. That is a deliberate trade-off for a
small standalone port; it is not a replacement for upstream when persistent indexing or
very low memory use is required.

## Install and use

```bash
pixi install
pixi run build
```

`pixi` activates `python/` automatically for every task. This self-contained example
creates a tiny FASTA file and runs unchanged with `pixi run python example.py`:

```python
import mojo_pyfastx as pyfastx
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "reads.fa"
    path.write_text(">contig example\nACGTNACGTN\n")
    fa = pyfastx.Fasta(path)
    print(len(fa), fa.gc_content)
    print(fa["contig"].seq)
    print(pyfastx.reverse_complement(fa["contig"].seq))
```

Run the full verification suite with:

```bash
pixi run build && pixi run test && pixi run bench
```

## Benchmarks

Measured with `pixi run bench` on Linux x86_64, Python 3.13.14. Numbers are best of
five runs on generated sequence data. The FASTA input has 5 million bases; the
initial-index case deletes upstream's `.fxi` before every measured run.

| case | mojo-pyfastx | pyfastx 2.3.1 | ratio | result |
| --- | ---: | ---: | ---: | --- |
| reverse_complement (5M bases) | 1.0 ms | 2.8 ms | 2.69x | faster |
| reverse_complement (32M bases) | 9.7 ms | 24.5 ms | 2.52x | faster |
| Fasta initial index + composition (5M bases) | 11.4 ms | 38.1 ms | 3.36x | faster |

The transform kernel uses byte SIMD with a scalar tail in a single pass. The
16-way split above 16 MiB is gone: the kernel is a table lookup and a store per
byte, well under two arithmetic operations per byte moved, so the split cost
more than it saved. The histogram has a SIMD fast path for uppercase DNA and a
scalar fallback for arbitrary bytes. The compact FASTA parser avoids per-line
objects and unnecessary single-record concatenation before composition. These
transforms and histograms move far less than roughly two arithmetic operations
per byte, so no GPU path is provided.

## How it works

The public classes parse FASTA/FASTQ records and maintain a name-to-record dictionary plus
an ordered record array for direct name and numeric access. Sequences and qualities are kept
as Python strings; slices create a lightweight `Sequence` view.

The histogram ctypes boundary passes integer addresses only. Mojo rebuilds those as
`UInt8` or `Int64` pointers and writes into Python-owned contiguous buffers. For ASCII
transforms, a small CPython API shim allocates the result string once and Mojo fills its
one-byte storage directly; the portable fallback uses a writable bytearray. A complement
lookup table supplies IUPAC complements, and the native byte histogram supplies nucleotide
composition and quality extrema. The Mojo kernels do not allocate or own Python memory.

## Parity testing

`pyfastx` is installed from PyPI by Pixi. The test suite compares covered
behavior directly against it, including IUPAC transforms, indexed access, statistics,
record properties, FASTA interval operations, streaming tuples, gzip input, and errors.
It also verifies the FFI byte histogram layout and its invalid-pointer guards.

MIT.
