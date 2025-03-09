Intended workflow and philosophy
================================

This tutorial describes the intended workflow the project was written for. By following this guide, you should be able to quickly start training models on your dataset without having to write any code. If this is too much abstraction for you and you prefer to write custom scripts, you can have a look at some minimal working code examples `here <./minimal.html>`_.

The following assumes you have already followed the `installation instructions <../index.html#installation>`_.

Recipes
-------

The project is structured as follows:

.. code-block:: bash

  .
  ├── mbchl/            # main package code
  ├── scripts/          # .py scripts to call from the command line
  ├── tests/            # unit tests
  ├── recipes/          # recipes for individual datasets
  ├── notebooks/        # .ipynb notebooks
  ├── data/             # data files e.g. speech corpora
  ├── lsf/              # HPC cluster scripts
  ├── docs/             # documentation
  ├── venv/             # virtual environment created with `python -m venv venv`
  ├── init_recipe.sh    # script to initialize a new recipe
  ├── pyproject.toml
  ├── README.md
  └── requirements.txt

The package code is intended to be used from within "recipes". A recipe is a folder that acts as a playground for a specific dataset or task. For example, the ``recipes/vbdemand`` recipe tackles speech enhancement on the VoiceBank+DEMAND dataset.

You can initialize a recipe with the following command:

.. code-block:: bash

  ./init_recipe.sh my_recipe

This does the following:

* A new folder ``recipes/my_recipe`` is created.
* Links pointing to ``data``, ``lsf``, ``scripts``, ``scripts/train.py`` and ``venv`` are placed in ``recipes/my_recipe``.
* An example configuration file is created and placed in ``recipes/my_recipe/models/example_model/config.yaml``. The configuration file contains model, dataset and training hyperparameters that are read by the training script.

Training a model
----------------

To train ``example_model``, first update the training and validation dataset paths in ``recipes/my_recipe/models/example_model/config.yaml``. Then, go to the recipe directory, activate the virtual environment and run the training script:

.. code-block:: bash

  cd recipes/my_recipe
  source venv/bin/activate
  python train.py models/example_model/

If training is successful, the following files are created inside ``models/example_model/`` next to the configuration file:

* ``checkpoints/``: Directory containing model checkpoints.
* ``log.log``: Log file.
* ``losses.npz``: NumPy file containing training losses, validation losses and validation metrics.
* ``training_curve.png``: Plot of training and validation losses.

Training is resumed if ``last.ckpt`` is detected in ``models/example_model/checkpoints/``. To train from scratch, delete the contents of ``models/example_model/checkpoints/`` or pass the ``-f`` option to ``train.py``. Note that using ``-f`` simply ignores ``last.ckpt``, which means that other checkpoints such as best checkpoints for each validation metric are not deleted. It is your responsibility to delete them to prevent the checkpoints directory from growing.

Configuration file
------------------

The default training script expects the following configuration fields:

* ``global_seed``: Random seed for reproducibility.
* ``ha``: Hearing aid name. See :attr:`~mbchl.has.HARegistry`.
* ``ha_kw``: Hearing aid keyword arguments. See the signature of the corresponding hearing aid.
* ``dataset.train``: Training dataset name. See :attr:`~mbchl.data.datasets.DatasetRegistry`.
* ``dataset.train_kw``: Training dataset keyword arguments. See the signature of the corresponding dataset.
* ``dataset.val``: Validation dataset name. See :attr:`~mbchl.data.datasets.DatasetRegistry`.
* ``dataset.val_kw``: Validation dataset keyword arguments. See the signature of the corresponding dataset. Can be a list of maps to use multiple validation datasets.
* ``trainer``: Trainer keyword arguments. See the signature of :class:`~mbchl.training.trainer.AudioTrainer`.

Configuration files can be added to version control. However this is the only file within each model directory that should be versioned. The model checkpoints and logs should never be versioned.

Overriding configuration file fields
------------------------------------

Configuration file fields can be overridden from the command line. This is useful for debugging without having to edit the configuration file. For example:

.. code-block:: bash

  python train.py models/example_model/ \
      trainer.use_wandb=False \
      trainer.device=cpu \
      trainer.workers=0 \
      trainer.val_period=1 \
      trainer.train_batch_sampler_kw.batch_size=1 \
      trainer.val_batch_sampler_kw.batch_size=1 \
      dataset.train_kw.n_files=1 \
      dataset.val_kw.n_files=1

Supported types are ``int``, ``float``, ``bool``, and ``str``. Fields that must be sequences cannot be overridden from the command line. See :func:`~mbchl.utils.parse_args`.

Weights and Biases
------------------

It is possible to log training metrics to `Weights and Biases <https://wandb.ai/site/>`_. To support this, create a ``.env`` file in your recipe directory with the following environment variables:

.. code-block:: bash

  WANDB_API_KEY="your_api_key"
  WANDB_PROJECT="project_name"
  WANDB_ENTITY="user_name_or_team_name"

Finally, to enable logging to Weights and Biases, set ``trainer.use_wandb=True`` in the model configuration file or from the command line. The ``.env`` file is loaded by the training script if it exists. Do not add ``.env`` to version control!

To resume a run in Weights and Biases, pass the ``--wandb_run_id`` option to the training script:

.. code-block:: bash

  python train.py models/example_model/ --wandb_run_id <run_id>
