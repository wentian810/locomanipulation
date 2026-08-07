# PHC source-only snapshot

This directory is the source/configuration/documentation portion of the PHC
working tree used by the server pipeline.  It intentionally excludes the
local Isaac Gym installation, downloaded training/evaluation data, sample
pickles, model checkpoints, cached outputs, and rendered media.  Run the
project setup documented in `README.MD` and the repository-level
`scripts/setup_phc_server_env.sh` to provision those external prerequisites.

The server copy had no `.git` metadata, so its exact upstream revision cannot
be asserted beyond the source snapshot and the upstream project reference in
`../patches/UPSTREAM_VERSIONS.md`.
