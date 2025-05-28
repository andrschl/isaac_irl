# isaac_reward_learning
Library for inverse reinforcement learning and RLHF for robotic manipulation.

## Collecting trajectories from existing policy
- `python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless`
- `python scripts/utils/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --num_demos 1000`
- `python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path 'logs/rsl_rl/franka_lift/demos/hdf_dataset.hdf5'`
- `~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/tools/inspect_demonstrations.py logs/rsl_rl/franka_lift/DATASET`
- `~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/tools/split_train_val.py PATH_TO_DATASET --ratio 0.2`
- `~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/train_bc.py --task Isaac-Lift-Cube-Franka-v0 --algo bc --dataset scripts/imitation_learning/logs/demos/hdf_dataset.hdf5`
- ``


## Connecting to the cluster
To connect to the cluster, you need to have SSH access. Use the following command to connect:
1. If you are a student: ask for access on the IZAR cluster to the service desk
2. ssh <gaspar_username>@izar.hpc.epfl.ch (recommend to use VSCode remote explorer tool)
3. (Optional but recommended) Setup a passwordless SSH connection by generating an SSH key pair and copying the public key to the cluster (Follow the SCITAS user guide)

## Using the cluster with SLURM
The cluster uses SLURM as the job scheduler, here are some basic commands to get you started:
- `squeue`: List all the jobs in the queue
- `Squeue`: List all **your** jobs in the queue
- `sinfo`: List all the nodes in the cluster
    - idle: "Means that the node is available for running jobs"
    - allocated or mix: "Means that the node is currently running a job"
- `sbatch`: Submit a job to the queue
- `scancel <job_id>`: Cancel a job in the queue
- `scancel --user <username>`: Cancel all your jobs in the queue


## Setup Isaac Lab with the cluster guide
Disclaimer: The following steps are taken from the [Isaac Lab Cluster Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html) & the [Isaac Lab Docker Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)
=> First perform the docker guide steps, then the cluster guide steps


### High-level of the workflow
1. Create a docker image of the Isaac Lab/Sim environment
2. Convert the Docker image to a Singularity image
3. Copy the Singularity image to the cluster
4. Run an experiment on the cluster:
   1. Copy a temporary version the Isaac Lab repository to the cluster (Try to keep the size of the repository small and ignore large files as it will inducesignificant overhead)
   2. Launch a SLURM job on the cluster to run the simulation on a compute node
   3. Copy the Singularity image & temp IsaacLab directory to a temporary directory on the compute node
   4. Run the Singularity image 
   5. Inside the Singularity container, install dependencies and run the python executable
   6. Copy the results (isaaclab/logs folder) back to the cluster's home directory



### Docker/Cluster files description
- `docker/cluster/.env.cluster`: Environment variables for the cluster
- `docker/cluster/cluster_interface.sh`: Script to be run from the local computer
- `docker/cluster/submit_job_slurm.sh`: Script to submit a job to the cluster
- `docker/cluster/run_singularity.sh`: Script that will be called on a compute node in order to run the Singularity image and the actual

### Cluster environment variables (`docker/cluster/.env.cluster` file)
These environment variables should not change from the first time you create the Cluster setup (except for `CLUSTER_PYTHON_EXECUTABLE` which can be modified accordingly):
- `CLUSTER_JOB_SCHEDULER`: SLURM
- `CLUSTER_ISAAC_SIM_CACHE_DIR`: Docker cache directory on the cluster
- `CLUSTER_ISAACLAB_DIR`: IsaacLab directory on the cluster where the results will be saved
- `CLUSTER_LOGIN`: Login node of the cluster (what you would write after `ssh` when connecting to the cluster)
- `CLUSTER_SIF_PATH`: Path to where the Singularity image will be stored on the cluster
- `REMOVE_CODE_COPY_AFTER_JOB`: If set to `true`, the temporary copy of the IsaacLab repository will be removed after the job is done
- `CLUSTER_PYTHON_EXECUTABLE`: The actual python executable to run the code inside the Singularity container

### Setup Instructions
1. One time setup on a computer meeting the Isaac Lab requirements:
   - Git clone the Isaac Lab repository
   - Install the correct Nvidia drivers
   - Install the correct version of Docker
   - Using the `docker/container.sh` script, create the Docker image and run it
   - Configure the `docker/cluster/.env.cluster` file
   - Run the `docker/cluster/cluster_interface.sh` script to convert the Docker image to a Singularity image
   - Copy the Singularity image to the cluster into a tar format 

2. From your local computer: 
   - Run the `docker/cluster/cluster_interface.sh` to run code directly on the cluster


## Steps you might need to add to the `run_singularity.sh` script
The Singularity Image is a static image that does not change. Hence it's state is fixed to it's creation time. Here are commands you might need to add to the `run_singularity.sh` script before running your Python code:
- Install the required python packages inside the Singularity container
- Set environment variables for the Singularity container....
- Reinstalling the Isaac Lab Task package of you added new tasks

## Running a training on the cluster
1. Connect to the EPFL VPN if your are not on campus
2. Run `./docker/cluster/cluster_interface.sh job --task Isaac-Ant-Direct-v0 --headless`
3. Once your repository is copied to the cluster, you will be notified of a scheduled SLURM Job
4. View job status using `Squeue`
5. View job outputs in the generated .out file (you can specify the output file name in the `docker/cluster/submit_job_slurm.sh` script with `#SBATCH --output="/home/<username>/logs/%j--$4--%x.out"` for better logging
)

## Vizualisation Training
1. Connect to the cluster using `ssh -X <username>@<cluster_login>` to enable X11 forwarding
2. Download the Isaaclab WebRTC latest release directly on the cluster using `wget`
3. Give `chmod +x` to the `webrtc` binary
4. Run a training and add `--livestream 2` as argument to the training script
5. Find the IP adress of the compute where the training is running using
   1. Use `Squeue` to find the node name
   2. Use `ping <node_name>` to find the IP address
6. Launch the WebRTC binary
7. Connect to the IP address of the compute on the WebRTC application