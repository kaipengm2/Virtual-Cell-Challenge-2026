# External data and software

The MIT license in this repository applies to the implementation and its
documentation. It does not relicense external data, annotations, libraries, or the
official challenge service. No biological matrices or third-party source trees
are redistributed in this repository.

| Source | Use | Primary reference |
|---|---|---|
| Replogle et al. 2022 K562 GWPS | Raw counts and gem-group-matched controls | [Publisher data record](https://figshare.com/articles/dataset/20029387) |
| X-Atlas-Orion | HCT116 / HEK293T expression and sample metadata | [Dataset repository](https://huggingface.co/datasets/slaf-project/X-Atlas-Orion) |
| Arc Virtual Cell Challenge 2025 | Released H1 train, validation and test observations | [Arc 2026 challenge announcement](https://arcinstitute.org/news/virtual-cell-challenge-2026); exact GCS object URLs in `docs/raw-assets.json` |
| Genome-scale T-cell Perturb-seq | CD4 publisher DESeq2 effects and quality flags | [Publisher analysis repository](https://github.com/emdann/GWT_perturbseq_analysis_2025), [data guide](https://github.com/emdann/GWT_perturbseq_analysis_2025/blob/master/metadata/data_sharing_readme.md) |
| GENCODE v47 | Gene-boundary coordinates for promoter-neighbor prior | [Release 47](https://www.gencodegenes.org/human/release_47.html) |
| VCC 2026 controls and CLI | Destination control populations, format validation, packaging | [Official CLI documentation](https://vcc-cli-wiki.virtualcellchallenge.org/) |
| Forrest Sheldon H1 benchmark v0.2.0 | Original H1 training asset registry and historical development evaluation | [Benchmark repository](https://github.com/forrestsheldon/vcc2026-h1-benchmark) |

The H1 benchmark package is not required by this inference implementation and its
code is not vendored. The historical H1 training object hash is retained in the
asset manifest. Public challenge score definitions belong to the official
evaluation system; this repository does not reimplement or claim to reproduce
hidden server scoring.

Core inference uses NumPy, SciPy, pandas, AnnData and h5py. Optional extraction
uses SLAF/Lance and Arrow; optional compression uses Numba. These are installed
as dependencies under their own licenses. The optional mapped prep adapter
invokes the locally installed VCC CLI module and contains no copied CLI implementation.

Before separately redistributing downloaded or derived datasets, establish the
applicable permissions with the provider. Public availability and a recorded
checksum alone do not establish redistribution rights.
