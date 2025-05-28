#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    --nv --writable --containall $TMPDIR/$2.sif \
    bash -c "
        export ISAACLAB_PATH=/workspace/isaaclab &&
        cd /workspace/isaaclab &&
        /isaac-sim/python.sh -m pip uninstall -y toml &&
        /isaac-sim/python.sh -m pip install toml &&
        /isaac-sim/python.sh -m pip show toml &&      
        /isaac-sim/python.sh -m pip install --upgrade pip &&
        /isaac-sim/python.sh -m pip install --upgrade pip setuptools wheel &&
        /isaac-sim/python.sh -m pip install -e source/isaaclab_tasks &&
        export WANDB_USERNAME=sebastien-epfl-epfl &&
        export WANDB_API_KEY=377aa0f3fcfdedeeed6ad9a746b76bb67204b0e9 &&
        echo $TMPDIR &&
        export ACCEPT_EULA=Y &&
        /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}
    "
        # export ACCEPT_EULA=Y &&
        # /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}
        # /isaac-sim/runheadless.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}

        # /isaac-sim/runheadless.sh
# copy resulting cache files back to host
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
