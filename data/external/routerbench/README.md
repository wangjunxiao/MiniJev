# RouterBench import

The raw RouterBench files are not vendored. Download a JSONL, JSON, CSV or
Parquet export from the
[MIT-licensed RouterBench dataset](https://huggingface.co/datasets/withmartian/routerbench)
and convert it locally:

```powershell
$env:PYTHONPATH = "D:\jev\tinygrad;D:\jev\MiniJev"
python -m minijev.convert routerbench --data path\routerbench.jsonl --audit
python -m minijev.convert routerbench --data path\routerbench.jsonl `
  --output routerbench_v1.jsonl --quality-threshold 0.8
```

The converter derives labels from the cheapest model meeting the quality
threshold. It is a declared cost/quality policy, not a label supplied by
RouterBench. Every output row contains a counterfactual candidate table and the
threshold used to derive its route.
