# Music Atlas settings for WCSS. Copy to ~/.atlas-env.sh, set ATLAS_ACCOUNT and
# `source ~/.atlas-env.sh` after each login; sbatch passes the variables to jobs.

# The music service's SLURM account (see `PD-info` or the e-science.pl portal).
# Accounts with several services default to another grant, so set it here.
export ATLAS_ACCOUNT=YOUR_ACCOUNT
export SBATCH_ACCOUNT=$ATLAS_ACCOUNT
export SALLOC_ACCOUNT=$ATLAS_ACCOUNT

export ATLAS_STORAGE=/lustre/pd03/$ATLAS_ACCOUNT/music-atlas
export ATLAS_DATA_DIR=$ATLAS_STORAGE/data
export ATLAS_ARTIFACTS_DIR=$ATLAS_STORAGE/artifacts
export ATLAS_OUTPUTS_DIR=$ATLAS_STORAGE/outputs
export UV_CACHE_DIR=$ATLAS_ARTIFACTS_DIR/uv-cache
export HF_HOME=$ATLAS_ARTIFACTS_DIR/hf-cache
export UV_PROJECT_ENVIRONMENT=$ATLAS_ARTIFACTS_DIR/venv-$USER
export PATH=$HOME/.local/bin:$PATH

# Packages installed with `pip install --user` would shadow the project environment.
export PYTHONNOUSERSITE=1
# New files on PD stay writable for everyone on the service.
umask 0002
