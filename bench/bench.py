"""Measure the Mojo kernels against pyfastx on the same generated inputs."""

from __future__ import annotations

import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pyfastx

import mojo_pyfastx as mpf


def best(fn, repeat: int = 5) -> float:
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        result = min(result, time.perf_counter() - start)
    return result


def main() -> None:
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(np.array(list("ACGTN")), size=5_000_000))
    parallel_seq = (seq * 7)[:32_000_003]
    with tempfile.TemporaryDirectory() as directory:
        fasta = Path(directory) / "large.fa"
        fasta.write_text(">large generated\n" + "\n".join(seq[i:i + 100] for i in range(0, len(seq), 100)) + "\n")
        mpf.reverse_complement(seq)
        pyfastx.reverse_complement(seq)
        mpf.reverse_complement(parallel_seq)
        pyfastx.reverse_complement(parallel_seq)
        mpf.Fasta(fasta)
        pyfastx.Fasta(str(fasta))

        def ours_index():
            return mpf.Fasta(fasta).composition

        def theirs_index():
            index = Path(f"{fasta}.fxi")
            if index.exists():
                index.unlink()
            return pyfastx.Fasta(str(fasta)).composition

        cases = [
            ("reverse_complement (5M bases)", lambda: mpf.reverse_complement(seq),
             lambda: pyfastx.reverse_complement(seq)),
            ("reverse_complement (32M bases)", lambda: mpf.reverse_complement(parallel_seq),
             lambda: pyfastx.reverse_complement(parallel_seq)),
            ("Fasta initial index + composition (5M bases)", ours_index, theirs_index),
        ]
        print(f"Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}")
        print("| case | mojo-pyfastx | pyfastx | ratio | result |")
        print("| --- | ---: | ---: | ---: | --- |")
        for name, ours, theirs in cases:
            a, b = best(ours), best(theirs)
            verdict = "faster" if a < b else "slower"
            print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {b / a:.2f}x | {verdict} |")


if __name__ == "__main__":
    main()
