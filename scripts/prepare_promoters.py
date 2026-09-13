"""Build the v38 gene-boundary TSS prior from pinned GENCODE v47 coordinates."""

import argparse
import gzip
from pathlib import Path
import re

import pandas as pd

from prepare_common import verified_asset
from vcc_atlas.io import axis_csv


def pairs(table, targets, genes):
    unique = table[~table.gene.duplicated(keep=False)].set_index("gene")
    relevant = unique.loc[unique.index.intersection(genes)]
    rows = []
    for target in targets:
        if target not in unique.index:
            continue
        a = unique.loc[target]
        near = relevant[
            (relevant.chromosome == a.chromosome) & ((relevant.tss - a.tss).abs() <= 5000)
        ]
        for name, b in near.iterrows():
            if name == target:
                continue
            divergent = a.strand != b.strand and (
                (a.strand == "+" and a.tss > b.tss) or (a.strand == "-" and a.tss < b.tss)
            )
            rows.append(
                dict(
                    target=target,
                    neighbor=name,
                    distance=abs(int(a.tss) - int(b.tss)),
                    chromosome=a.chromosome,
                    target_tss=int(a.tss),
                    neighbor_tss=int(b.tss),
                    target_strand=a.strand,
                    neighbor_strand=b.strand,
                    divergent=divergent,
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "target",
            "neighbor",
            "distance",
            "chromosome",
            "target_tss",
            "neighbor_tss",
            "target_strand",
            "neighbor_strand",
            "divergent",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--official-targets", type=Path, required=True)
    parser.add_argument("--official-genes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified_asset("gencode47", args.gtf)
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    with gzip.open(args.gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
            if "gene_name" not in attrs:
                continue
            rows.append(
                dict(
                    gene=attrs["gene_name"],
                    chromosome=fields[0],
                    strand=fields[6],
                    tss=int(fields[3] if fields[6] == "+" else fields[4]),
                )
            )
    result = pairs(
        pd.DataFrame(rows),
        axis_csv(args.official_targets, "target_gene"),
        axis_csv(args.official_genes, "gene_name"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} promoter pairs")


if __name__ == "__main__":
    main()
