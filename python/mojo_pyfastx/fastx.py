"""pyfastx-compatible record objects over a compact in-memory index.

The parser accepts plain and gzip-compressed FASTA/FASTQ.  It keeps one Python
string per sequence/quality and delegates bytewise transforms and histograms to
Mojo, avoiding Python character loops on large records.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from statistics import median
from typing import Callable, Iterator, Sequence as TypingSequence

import numpy as np

from ._lib import histogram, transform


def gzip_check(file_name: str | Path) -> bool:
    with open(file_name, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def reverse_complement(seq: str) -> str:
    return transform(seq, 2)


def _read_text(file_name: str | Path) -> tuple[str, bool]:
    path = Path(file_name)
    if not path.is_file():
        raise FileExistsError(str(path))
    compressed = gzip_check(path)
    opener = gzip.open if compressed else open
    with opener(path, "rt", newline="") as handle:
        return handle.read(), compressed


def _header_name(header: str) -> str:
    return header.split(None, 1)[0] if header else ""


def _counts_dict(counts: np.ndarray) -> dict[str, int]:
    return {chr(i): int(n) for i, n in enumerate(counts) if n}


def _record_histogram(records: list["_Record"], attribute: str) -> np.ndarray:
    if len(records) == 1:
        return histogram(getattr(records[0], attribute))
    return histogram("".join(getattr(record, attribute) or "" for record in records))


def _nucleotide_counts(value: str) -> tuple[int, int, int, int]:
    counts = histogram(value)
    return tuple(int(counts[ord(base)] + counts[ord(base.lower())]) for base in "ACGT")


def _gc_content(value: str) -> float:
    a, c, g, t = _nucleotide_counts(value)
    total = a + c + g + t
    return 0.0 if total == 0 else (c + g) * 100.0 / total


def _gc_skew(value: str) -> float:
    _, c, g, _ = _nucleotide_counts(value)
    return 0.0 if c + g == 0 else (g - c) / (g + c)


@dataclass(frozen=True)
class _Record:
    name: str
    description: str
    seq: str
    raw: str
    ident: int
    qual: str | None = None


class Sequence:
    """A FASTA record, including cheap random slices of its indexed sequence."""

    def __init__(self, record: _Record, start: int = 1, end: int = 0):
        self._record, self.start, self.end = record, start, end

    @property
    def id(self) -> int:
        return self._record.ident

    @property
    def name(self) -> str:
        if self.end:
            return f"{self._record.name}:{self.start}-{self.end}"
        return self._record.name

    @property
    def description(self) -> str:
        return self._record.description

    @property
    def seq(self) -> str:
        return self._record.seq

    @property
    def raw(self) -> str:
        return self._record.raw

    @property
    def reverse(self) -> str:
        return transform(self.seq, 1)

    @property
    def complement(self) -> str:
        return transform(self.seq, 0)

    @property
    def antisense(self) -> str:
        return reverse_complement(self.seq)

    @property
    def composition(self) -> dict[str, int]:
        return _counts_dict(histogram(self.seq))

    @property
    def gc_content(self) -> float:
        return _gc_content(self.seq)

    @property
    def gc_skew(self) -> float:
        return _gc_skew(self.seq)

    def search(self, subseq: str) -> int:
        return self.seq.find(subseq) + 1

    def __len__(self) -> int:
        return len(self.seq)

    def __str__(self) -> str:
        return self.seq

    def __repr__(self) -> str:
        if self.end:
            return f"<Sequence> {self.name}"
        return f"<Sequence> {self.name} with length of {len(self)}"

    def __contains__(self, value: str) -> bool:
        return value in self.seq

    def __getitem__(self, item: int | slice) -> str | "Sequence":
        if isinstance(item, int):
            return self.seq[item]
        if not isinstance(item, slice):
            raise TypeError("sequence indices must be integers or slices")
        if item.step not in (None, 1):
            raise ValueError("slice step cannot > 1")
        lo, hi, _ = item.indices(len(self))
        child = self.seq[item]
        record = _Record(self._record.name, self.description, child, self.raw, self.id)
        return Sequence(record, self.start + lo, self.start + hi)

    def __iter__(self) -> Iterator[str]:
        if self.end:
            raise RuntimeError("could not iterate a subsequence")
        body = self.raw.splitlines()[1:]
        yield from (line.strip() for line in body if line.strip())


class Read:
    """A FASTQ record with sequence transforms and Phred quality conversion."""

    def __init__(self, record: _Record, phred: int = 33):
        self._record, self._phred = record, phred

    @property
    def id(self) -> int:
        return self._record.ident

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def description(self) -> str:
        return self._record.description

    @property
    def seq(self) -> str:
        return self._record.seq

    @property
    def qual(self) -> str:
        return self._record.qual or ""

    @property
    def quali(self) -> list[int]:
        return [ord(value) - self._phred for value in self.qual]

    @property
    def raw(self) -> str:
        return self._record.raw

    @property
    def reverse(self) -> str:
        return transform(self.seq, 1)

    @property
    def complement(self) -> str:
        return transform(self.seq, 0)

    @property
    def antisense(self) -> str:
        return reverse_complement(self.seq)

    def __len__(self) -> int:
        return len(self.seq)

    def __str__(self) -> str:
        return self.seq

    def __repr__(self) -> str:
        return f"<Read> {self.name} with length of {len(self)}"

class _IndexedFile:
    _kind = "record"

    def _lookup(self, key: int | str):
        if not self._indexed:
            raise RuntimeError("random access requires build_index=True")
        if isinstance(key, int):
            try:
                return self._records[key]
            except IndexError:
                raise IndexError("record index out of range") from None
        if isinstance(key, str):
            try:
                return self._by_name[key]
            except KeyError:
                raise KeyError(key) from None
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._by_name

    def keys(self) -> TypingSequence[str]:
        return list(self._by_name)

    def build_index(self) -> None:
        self._indexed = True


class Fasta(_IndexedFile):
    """Index FASTA records in memory and expose pyfastx-style accessors."""

    _kind = "sequence"

    def __init__(self, file_name, build_index: bool = True, full_name: bool = False,
                 uppercase: bool = False, key_func: Callable[[str], str] | None = None,
                 full_index: bool = False):
        if key_func is not None and not callable(key_func):
            raise TypeError("key_func must be callable")
        self.file_name = str(file_name)
        text, self.is_gzip = _read_text(file_name)
        self._full_name, self._uppercase = full_name, uppercase
        self._records = _parse_fasta(text, uppercase, key_func)
        self._by_name = {record.name: record for record in self._records}
        self._indexed = build_index
        self._all_counts = _record_histogram(self._records, "seq")

    def __repr__(self) -> str:
        suffix = f" contains {len(self)} sequences" if self._indexed else ""
        return f"<Fasta> {self.file_name}{suffix}"

    def __iter__(self):
        if self._indexed:
            yield from (Sequence(record) for record in self._records)
        else:
            for record in self._records:
                name = record.description if self._full_name else record.name
                yield name, record.seq

    def __getitem__(self, key: int | str) -> Sequence:
        return Sequence(self._lookup(key))

    @property
    def size(self) -> int:
        return sum(map(len, (record.seq for record in self._records)))

    @property
    def composition(self) -> dict[str, int]:
        return _counts_dict(self._all_counts)

    @property
    def gc_content(self) -> float:
        return _gc_content("".join(record.seq for record in self._records))

    @property
    def gc_skew(self) -> float:
        return _gc_skew("".join(record.seq for record in self._records))

    @property
    def type(self) -> str:
        letters = set("".join(record.seq.upper() for record in self._records))
        if letters <= set("ACGTNRYSWKMBDHV-"):
            return "DNA"
        if letters <= set("ACGUNRYSWKMBDHV-"):
            return "RNA"
        return "protein"

    @property
    def longest(self) -> Sequence:
        if not self._records:
            raise RuntimeError("could not found longest sequence")
        return Sequence(max(self._records, key=lambda record: len(record.seq)))

    @property
    def shortest(self) -> Sequence:
        if not self._records:
            raise RuntimeError("not found shortest sequence")
        return Sequence(min(self._records, key=lambda record: len(record.seq)))

    @property
    def mean(self) -> float:
        return self.size / len(self._records) if self._records else 0.0

    @property
    def median(self) -> float:
        return median(len(record.seq) for record in self._records) if self._records else 0.0

    def count(self, min_length: int) -> int:
        return sum(len(record.seq) >= min_length for record in self._records)

    def nl(self, percentage: int) -> tuple[int, int]:
        if not 0 < percentage <= 100:
            raise ValueError("percentage must be in 1..100")
        lengths = sorted((len(record.seq) for record in self._records), reverse=True)
        threshold, total = sum(lengths) * percentage / 100.0, 0
        for index, length in enumerate(lengths, 1):
            total += length
            if total >= threshold:
                return length, index
        return 0, 0

    def fetch(self, name: str, interval):
        seq = self[name].seq
        if isinstance(interval, tuple) and len(interval) == 2 and all(isinstance(v, int) for v in interval):
            intervals = [interval]
        elif isinstance(interval, list):
            intervals = interval
        else:
            raise ValueError("interval must be a (start, end) tuple or list of tuples")
        pieces = []
        for item in intervals:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("each interval must be a (start, end) tuple")
            start, end = item
            if start < 1 or end < start:
                raise ValueError("invalid interval")
            pieces.append(seq[start - 1:end])
        return "".join(pieces)

    def flank(self, name: str, start: int, end: int, flank_length: int = 50,
              use_cache: bool = False) -> tuple[str, str]:
        seq = self[name].seq
        return seq[max(0, start - flank_length - 1):start - 1], seq[end:end + flank_length]


class Fastq(_IndexedFile):
    """Index standard FASTQ records in memory and preserve pyfastx names."""

    _kind = "read"

    def __init__(self, file_name, build_index: bool = True, full_name: bool = False):
        self.file_name = str(file_name)
        text, self.is_gzip = _read_text(file_name)
        self._full_name = full_name
        self._records = _parse_fastq(text)
        self._by_name = {record.name: record for record in self._records}
        self._indexed = build_index
        self._all_counts = _record_histogram(self._records, "seq")
        self._quality_counts = _record_histogram(self._records, "qual")
        values = np.flatnonzero(self._quality_counts)
        self.minqual = int(values[0]) if values.size else 0
        self.maxqual = int(values[-1]) if values.size else 0
        self.phred = 33 if self.minqual < 59 else 64

    def __repr__(self) -> str:
        suffix = f" contains {len(self)} reads" if self._indexed else ""
        return f"<Fastq> {self.file_name}{suffix}"

    def __iter__(self):
        if self._indexed:
            yield from (Read(record, self.phred) for record in self._records)
        else:
            for record in self._records:
                name = record.description[1:] if self._full_name else record.name
                yield name, record.seq, record.qual

    def __getitem__(self, key: int | str) -> Read:
        return Read(self._lookup(key), self.phred)

    @property
    def size(self) -> int:
        return sum(map(len, (record.seq for record in self._records)))

    @property
    def composition(self) -> dict[str, int]:
        return _counts_dict(self._all_counts)

    @property
    def gc_content(self) -> float:
        return _gc_content("".join(record.seq for record in self._records))

    @property
    def avglen(self) -> float:
        return self.size / len(self._records) if self._records else 0.0

    @property
    def minlen(self) -> int:
        return min((len(record.seq) for record in self._records), default=0)

    @property
    def maxlen(self) -> int:
        return max((len(record.seq) for record in self._records), default=0)

    @property
    def encoding_type(self) -> list[str]:
        if not self.maxqual:
            return ["Unknown"]
        result = []
        if 33 <= self.minqual and self.maxqual <= 73:
            result.append("Sanger Phred+33")
        if 33 <= self.minqual and self.maxqual <= 74:
            result.append("Illumina 1.8+ Phred+33")
        if 59 <= self.minqual and self.maxqual <= 104:
            result.append("Solexa Solexa+64")
        if 64 <= self.minqual and self.maxqual <= 104:
            result.append("Illumina 1.3+ Phred+64")
        if 66 <= self.minqual and self.maxqual <= 104:
            result.append("Illumina 1.5+ Phred+64")
        if 33 <= self.minqual and self.maxqual <= 126:
            result.append("PacBio HiFi Phred+33")
        return result or ["Unknown"]


class Fastx:
    """Streaming tuple interface that detects FASTA versus FASTQ automatically."""

    def __init__(self, file_name, format: str | None = None, comment: bool = False,
                 uppercase: bool = False):
        self.file_name = str(file_name)
        text, self.is_gzip = _read_text(file_name)
        inferred = "fasta" if text.lstrip().startswith(">") else "fastq"
        self.format = (format or inferred).lower()
        if self.format not in {"fasta", "fastq"}:
            raise RuntimeError("format must be fasta or fastq")
        self.comment, self.uppercase = comment, uppercase
        self._records = _parse_fasta(text, uppercase, None) if self.format == "fasta" else _parse_fastq(text)

    def __repr__(self) -> str:
        return f"<Fastx> {self.format} {self.file_name}"

    def __iter__(self):
        for record in self._records:
            comment = record.description.split(None, 1)[1] if len(record.description.split(None, 1)) == 2 else ""
            if self.format == "fasta":
                yield (record.name, record.seq, comment) if self.comment else (record.name, record.seq)
            else:
                yield (record.name, record.seq, record.qual, comment) if self.comment else (record.name, record.seq, record.qual)


def _parse_fasta(text: str, uppercase: bool, key_func: Callable[[str], str] | None) -> list[_Record]:
    if text.startswith(">"):
        records = _parse_fasta_compact(text, uppercase, key_func)
        if records is not None:
            return records
    return _parse_fasta_lines(text, uppercase, key_func)


def _parse_fasta_compact(text: str, uppercase: bool,
                         key_func: Callable[[str], str] | None) -> list[_Record] | None:
    chunks = text.split("\n>")
    records = []
    for index, chunk in enumerate(chunks):
        raw = chunk if index == 0 else ">" + chunk
        header_end = raw.find("\n")
        if header_end < 0:
            header, body = raw, ""
        else:
            header, body = raw[:header_end], raw[header_end + 1:]
        if "\t" in body or " " in body or "\v" in body or "\f" in body:
            return None
        if index + 1 < len(chunks):
            raw += "\n"
        description = header[1:].strip()
        name = key_func(description) if key_func else _header_name(description)
        seq = body.replace("\n", "").replace("\r", "")
        records.append(_Record(name, description, seq.upper() if uppercase else seq, raw, index + 1))
    return records


def _parse_fasta_lines(text: str, uppercase: bool,
                       key_func: Callable[[str], str] | None) -> list[_Record]:
    records, header, body, raw, ident = [], None, [], [], 0
    for line in text.splitlines(keepends=True):
        if line.startswith(">"):
            if header is not None:
                ident += 1
                description = header[1:].strip()
                name = key_func(description) if key_func else _header_name(description)
                seq = "".join(body)
                records.append(_Record(name, description, seq.upper() if uppercase else seq, "".join(raw), ident))
            header, body, raw = line, [], [line]
        elif header is not None:
            body.append(line.strip())
            raw.append(line)
    if header is not None:
        ident += 1
        description = header[1:].strip()
        name = key_func(description) if key_func else _header_name(description)
        seq = "".join(body)
        records.append(_Record(name, description, seq.upper() if uppercase else seq, "".join(raw), ident))
    if not records:
        raise RuntimeError("not a FASTA file")
    return records


def _parse_fastq(text: str) -> list[_Record]:
    lines, records, index, ident = text.splitlines(keepends=True), [], 0, 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        header_at = index
        if not lines[index].startswith("@"):
            raise RuntimeError("not a FASTQ file")
        header = lines[index].rstrip("\r\n")
        index += 1
        seq_lines = []
        while index < len(lines) and not lines[index].startswith("+"):
            seq_lines.append(lines[index])
            index += 1
        if index == len(lines):
            raise RuntimeError("FASTQ sequence has no quality separator")
        index += 1
        seq = "".join(line.strip() for line in seq_lines)
        quality_lines, quality_length = [], 0
        while index < len(lines) and quality_length < len(seq):
            quality_lines.append(lines[index])
            quality_length += len(lines[index].strip())
            index += 1
        qual = "".join(line.strip() for line in quality_lines)
        if len(qual) != len(seq):
            raise RuntimeError("FASTQ sequence and quality have different lengths")
        ident += 1
        description = header
        records.append(_Record(_header_name(header[1:]), description, seq,
                               "".join(lines[header_at:index]), ident, qual))
    if not records:
        raise RuntimeError("not a FASTQ file")
    return records
