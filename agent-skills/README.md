# Graph2Mat Agent Skills v2

Updated skills with four refinements:

1. Never treat dH/dR as a sufficient proxy for EPC.
2. Separate sample efficiency from total electronic-structure labeling cost.
3. Do not impose a rigid universal seed count; replication depends on effect size and variance.
4. For sampler comparisons, separate budget matching from physical-distribution matching.

Install by copying these directories into your repository's `agent-skills/` directory, then run:

```bash
python3 agent-skills/validate_skills.py
./agent-skills/sync.sh
```

The sync script only manages these six skill directories.
