#!/usr/bin/env python3
#SBATCH -t 04:00:00
#SBATCH -p short 
#SBATCH -o logs/downsample.o%j
#SBATCH -e logs/downsample.e%j
#SBATCH -J downsample_cloudvolume
#SBATCH --mail-type=END
#SBATCH --mem 32GB

import sys

import igneous.task_creation as tc
from cloudvolume.lib import Bbox


# --- Configurations --- #
# Change this path to point to an existing cloudvolume you want to downsample
cloud_path = 'gs://your-bucket/your-dataset'

# If you want to downsample the whole cloudvolume, leave "bounds = None"
bounds = None
# If you want to downsample only a subset of the dataset, set the bounds like this:
#bounds = Bbox([0, 0, 100], [23358, 14102, 150])  # Units are in voxels

# Choose how many levels you want to downsample your dataset
num_mips = 4

# Choose whether to downsample just in x and y (the default - appropriate for
# anisotropic data like serial section EM data)
#factor = (2, 2, 1)
# Or downsample in x, y, and z (appropriate for isotropic data, especially if
# the data is already stored in cube-shaped chunks, which Xray datasets often
# are but EM datasets usually aren't)
factor = (2, 2, 2)

# --- End configurations --- #


def show_help():
    print("""\
Generate downsampled versions (also called "mipmaps" or an "image pyramid") of a cloudvolume.

First, open this script and set values in the "Configurations" section at the top

Second, generate tasks by running:
  ./downsample.py create_task_queue
This will generate a queue of jobs to run, stored in the folder "downsampling_tasks"

Third, run the downsampling tasks in one of a few ways.
Option 1 - Submit as a job to the o2 cluster by running:
  sbatch downsample.py run_tasks_from_queue
You can run this sbatch command multiple times to get multiple downsampling jobs running, all
pulling from the same job queue specified by the files in the "downsampling_tasks" folder.
Depending on how many tasks you have, submitting anywhere from 1 to 20 or so times is reasoanble.
Option 2 - Run tasks directly while logged into a lab server (e.g. gandalf, htem, radagast):
  ./downsample.py run_tasks_from_queue
The line above will run just a single thread. If you want to run in parallel using many threads, run something like:
  for i in {1..8}; do ./downsample.py run_tasks_from_queue & done
If you are downsampling large EM datasets, be careful not to run so many threads that you use up all the memory on the server
    """)

queuepath = 'igneous_tasks'

def create_task_queue():
    from taskqueue import TaskQueue
    tq = TaskQueue('fq://'+queuepath)
    tasks = tc.create_downsampling_tasks(
        cloud_path,
        mip=0,             # Starting mip
        num_mips=num_mips, # Final mip to downsample to
        bounds=bounds,
        factor=factor,
        fill_missing=True
    )
    tq.insert(tasks)
    print('Done adding {} tasks to queue at {}'.format(len(tasks), queuepath))


def run_tasks_from_queue():
    from taskqueue import TaskQueue
    tq = TaskQueue('fq://'+queuepath)
    print('Working on tasks from filequeue "{}"'.format(queuepath))
    tq.poll(
        verbose=True, # prints progress
        lease_seconds=7200,
        tally=True # makes tq.completed work, logs 1 byte per completed task
    )
    print('Done')


def run_tasks_locally(n_cores=16):
    from taskqueue import LocalTaskQueue
    tq = LocalTaskQueue(parallel=n_cores)
    tasks = tc.create_downsampling_tasks(
        cloud_path,
        mip=0,             # Starting mip
        num_mips=num_mips, # Final mip to downsample to
        bounds=bounds,
        factor=factor,
        fill_missing=True
    )
    tq.insert(tasks)
    print('Running in-memory task queue on {} cores'.format(n_cores))
    tq.execute()
    print('Done')


if __name__ == '__main__':
    l = locals()
    public_functions = [f for f in l if callable(l[f]) and f[0] != '_']
    if len(sys.argv) <= 1 or not sys.argv[1] in public_functions:
        show_help()
    else:
        func = l[sys.argv[1]]
        args = []
        kwargs = {}
        for arg in sys.argv[2:]:
            if '=' in arg:
                split = arg.split('=')
                kwargs[split[0]] = split[1]
            else:
                args.append(arg)
        func(*args, **kwargs)
