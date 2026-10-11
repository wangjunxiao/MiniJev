# Literary supplement

`literary_v1.jsonl` is a deterministic, controlled supplement for testing route
boundaries on short traditional/classical excerpts. It contains 12 works, four
requested operations per work, and three wording variants per semantic group.

The route is assigned from the operation:

- `local`: a bounded extraction task over the supplied text;
- `rag`: author or provenance lookup outside the supplied text;
- `cloud`: multi-step comparison or interpretation;
- `review`: copyright or legal-risk judgement.

The texts are included as short traditional/classical excerpts. The metadata
keeps the work, author, semantic group and a reminder to verify the rights of
the edition used before publication. This data is for controlled experiments,
not a human-labelled production benchmark.

Generate or regenerate it with:

```powershell
$env:PYTHONPATH = "D:\jev\tinygrad;D:\jev\MiniJev"
python -m minijev.literary --variants 3
```
