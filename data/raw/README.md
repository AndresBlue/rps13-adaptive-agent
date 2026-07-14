# Raw datasets

Place external sequential rock-paper-scissors datasets here. Examples include
Brockbank RPS, hm_rps_public, or any local CSV with human/agent move sequences.

The project does not download data automatically. Use
`rps13.data.dataset_loader.normalize_rps_dataset()` with a column mapping to
convert an external CSV into the internal format used by training.
