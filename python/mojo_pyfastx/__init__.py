"""A Mojo-accelerated, pyfastx-shaped FASTA/FASTQ reader."""

from .fastx import Fasta, Fastq, Fastx, Read, Sequence, gzip_check, reverse_complement

__all__ = ["Fasta", "Fastq", "Fastx", "Read", "Sequence", "gzip_check", "reverse_complement", "version"]


def version() -> str:
    return "0.1.0"
