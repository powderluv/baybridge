# Third-Party DSL Samples

This directory holds metadata and generated outputs for an upstream sample corpus used to measure Baybridge compatibility.

Checked in:
- `MANIFEST.json`: upstream repo, pinned revision, and selected sample paths
- `.gitignore`: generated content that should not be committed

Generated on demand:
- `upstream/`: fetched sample files
- `FETCHED.txt`: fetch metadata
- `compatibility_matrix.json`
- `compatibility_matrix.md`

Fetch from an existing checkout:

```bash
python tools/fetch_thirdparty_samples.py --source-root /path/to/cutlass
```

Fetch by cloning the pinned upstream revision automatically:

```bash
python tools/fetch_thirdparty_samples.py
```

Analyze the fetched corpus:

```bash
python tools/analyze_thirdparty_samples.py
```
