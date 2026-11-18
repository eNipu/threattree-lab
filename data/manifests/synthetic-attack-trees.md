# Synthetic Attack Tree Dataset

- Dataset ID: `synthetic-attack-trees-v0.1`
- File: `data/curated/synthetic_attack_trees.jsonl`
- Format: JSON Lines, one reviewed-style SFT record per line
- Records: 6
- Created: 2026-11-18
- Source type: synthetic, fictional systems only
- Provenance: generated for ThreatTree Lab fine-tuning data preparation; no real organizations, incidents, customer data, or external MITRE snapshots are included
- Intended split: initial train candidate only; must be reviewed before use and must not be combined with final test scenarios
- Review status: `synthetic_unreviewed`
- Safety: scenarios are high-level and contain no exploit code, live-target instructions, or weaponized procedures
- Evidence references: use `synthetic:` identifiers unless a real source record and checksum are attached
- Notes: AI-drafted examples require human review, deduplication, schema validation, and graph validation before training or evaluation
