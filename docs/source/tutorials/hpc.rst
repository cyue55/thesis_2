Training models on the DTU HPC cluster
======================================

Submitting jobs
---------------

The DTU HPC cluster is managed with `IBM Spectrum LSF <https://www.ibm.com/docs/en/spectrum-lsf/>`_. Jobs are submitted by writing a job script and submitting it with the ``bsub`` command. The job script contains resource requests, environment setup and the commands to run the job. You can consult the `DTU HPC documentation <https://www.hpc.dtu.dk>`_ for more information about job submission and resources available behind each queue.

Manually writing job scripts for every training run can be cumbersome. To simplify this process, scripts that automate job submission are provided in the ``lsf`` directory. These scripts are meant to be called from the HPC login nodes.

First, SSH to the cluster using e.g. ``ssh <username>@login.hpc.dtu.dk``. Follow the `installation instructions <../index.html#installation>`_ to set up the project in the cluster (clone repository, create virtual environment and install requirements and project package). When creating the virtual environment, make sure to first load the right python version using e.g. ``module load python3/3.12.4`` before running ``python3 -m venv venv``. Once everything is set up, a training job can be submitted by first navigating to your recipe directory and then running the following command:

.. code-block:: bash

  ./lsf/train.sh models/example_model/

Multiple training jobs can be submitted by providing multiple model directories as arguments:

.. code-block:: bash

  ./lsf/train.sh models/first_model/ models/second_model/

Logs are written to ``lsf/logs/<job_id>.out`` and ``lsf/logs/<job_id>.err``.

By default, the job is submitted to the ``gpuv100`` queue, requesting 4 CPUs and 4 GB of memory for each CPU (i.e. 16 GB in total). Options passed to ``lsf/train.sh`` include:

- ``-q``: queue name
- ``-c``: number of CPUs
- ``-m``: memory per CPU
- ``-s``: selection string
- ``-w``: job dependency
- ``-W``: walltime
- ``-x``: extra arguments passed to ``train.py``

For example, to submit a job to the ``gpua100`` queue, requesting 8 CPUs, 8 GB of memory per CPU, and a 80 GB GPU, run:

.. code-block:: bash

  ./lsf/train.sh models/example_model/ -q gpua100 -c 8 -m 8 -s gpu80gb

The default walltime is 24 hours. The ``gpuv100`` and ``gpua100`` queues have a maximum walltime of 24 and 72 hours respectively. To request the maximum walltime on the ``gpua100`` queue, run:

.. code-block:: bash

  ./lsf/train.sh models/example_model/ -q gpua100 -W 72

You can schedule jobs with a dependency on another job with the ``-w`` option. For example, if training lasts longer than the walltime, you can schedule a job to resume training once the previous job has finished:

.. code-block:: bash

  ./lsf/train.sh models/example_model/ -w "ended(<job_id>)"

See the `LSF documentation <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=scheduling-dependency-conditions>`_ for information about the different dependency conditions.

To pass extra arguments to ``train.py``, use the ``-x`` option. For example, to use the ``--wandb_run_id`` option to resume a Weights and Biases run, use:

.. code-block:: bash

  ./lsf/train.sh models/example_model/ -x "--wandb_run_id <run_id>"

Monitoring jobs
---------------

The status of your jobs can be monitored with the LSF command ``bstat``:

.. code-block:: bash

  $ bstat
  JOBID      USER    QUEUE      JOB_NAME   NALLOC STAT  START_TIME      ELAPSED
  23295083   phigon  gpua100    jobname         8 RUN   Nov 29 06:20    3:00:13
  23295079   phigon  gpuv100    jobname         8 RUN   Nov 29 04:22    4:58:28
  23295080   phigon  gpuv100    jobname         8 RUN   Nov 29 04:35    4:45:05
  23295081   phigon  gpuv100    jobname         4 PEND       -          0:00:00
  23295084   phigon  gpua100    jobname         8 PEND       -          0:00:00
  23295085   phigon  gpua100    jobname         8 PEND       -          0:00:00
  23295086   phigon  gpua100    jobname         4 PEND       -          0:00:00

CPU usage can be monitored with the ``-C`` option:

.. code-block:: bash

  $ bstat -C
  JOBID      USER    QUEUE      JOB_NAME   NALLOC    ELAPSED     EFFIC
  23295083   phigon  gpua100    jobname         8    3:00:45     67.98
  23295079   phigon  gpuv100    jobname         8    4:59:00     69.96
  23295080   phigon  gpuv100    jobname         8    4:45:37     71.56

Memory usage can be monitored with the ``-M`` option:

.. code-block:: bash

  $ bstat -M
  JOBID      USER    QUEUE      JOB_NAME   NALLOC    MEM     MAX     AVG     LIM
  23295083   phigon  gpua100    jobname         8   12.7G   12.7G   11.2G   32.0G
  23295079   phigon  gpuv100    jobname         8   13.3G   13.5G   11.8G   32.0G
  23295080   phigon  gpuv100    jobname         8   13.5G   13.6G   11.9G   32.0G

GPU usage is less trivial to monitor and requires multiple LSF commands. The ``lsf/gpustat.sh`` script automates this:

.. code-block:: bash

  $ ./lsf/gpustat.sh 23295083

  HOST: n-62-12-22
  NGPUS NGPUS_SHARED_AVAIL NGPUS_EXCLUSIVE_AVAIL
  2     0                  0

  GPU_ID   MODE               MUSED     MRSV      TEMP   ECC    UT     MUT    PSTATE STATUS   ERROR
  1        EXCLUSIVE_PROCESS  32.9G     0M        52C    0      79%    15%    0      ok       -

Transfering data
----------------

Large data files are not meant to be stored in the login nodes, which only have 30 GB of storage per user. Instead, data files should be stored in a scratch directory. To be allocated space in the scratch directory, contact the HPC support.

Once you are allocated a scratch directory, e.g. ``/work3/<username>``, you can transfer data files from your local machine to the cluster using ``scp`` or ``rsync``. When doing so, use the ``transfer.gbar.dtu.dk`` node:

.. code-block:: bash

  scp -r path/to/local/data <username>@transfer.gbar.dtu.dk:/work3/<username>/path/to/remote/data

Or:

.. code-block:: bash

  rsync -zarv path/to/local/data <username>@transfer.gbar.dtu.dk:/work3/<username>/path/to/remote/data

Your training configuration files in the cluster should then point to the data files in the scratch directory. However, editing your configuration files in the cluster would introduce a mismatch with your local configuration files. Instead, you can set up symbolic links pointing to your data files in the scratch directory. For example, if the VoiceBank+DEMAND dataset is in ``/work3/<username>/vbdemand`` in the cluster, you can create a symbolic link in the project ``data`` directory in the cluster using:

.. code-block:: bash

  ln -s /work3/<username>/vbdemand data/vbdemand

To view your disk usage in your home directory on the cluster, use the ``getquota_zhome.sh`` command. To view your disk usage in your ``/work3`` scratch directory, use the ``getquota_work3.sh`` command.
