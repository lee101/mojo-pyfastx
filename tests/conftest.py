from __future__ import annotations

import gzip

import pytest


@pytest.fixture()
def files(tmp_path):
    fasta = tmp_path / "reads.fa"
    fasta.write_text(">alpha description one\nACGTNacgtn\nTT\n>beta second\nGGCC\n")
    fastq = tmp_path / "reads.fq"
    fastq.write_text("@r1 first read\nACGTN\n+\nII!5?\n@r2 second\nGG\n+\n##\n")
    with gzip.open(f"{fasta}.gz", "wt") as handle:
        handle.write(fasta.read_text())
    with gzip.open(f"{fastq}.gz", "wt") as handle:
        handle.write(fastq.read_text())
    return fasta, fastq
