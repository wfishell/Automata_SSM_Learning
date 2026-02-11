# Automata_SSM_Learning

## Setup Instructions

### Prerequisites

- Docker installed with NVIDIA Container Toolkit (for GPU support)
- VS Code with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`)

### 1. Build the Docker Image

From the project root directory:

```bash
docker build -f .devcontainer/Learning.dockerfile -t learning:latest .
```

### 2. Launch the Dev Container

1. Open the project folder in VS Code.
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select **Dev Containers: Reopen in Container**.
3. VS Code will start the container using the pre-built `learning:latest` image and apply the settings from `.devcontainer/devcontainer.json`.

### 3. Install Additional Python Packages

Once inside the dev container, open a terminal and run:

```bash
pip install matplotlib pydot scikit-learn pandas
```

### 4. Run the Warm-Start Drift Experiment

The `drift_test_pipeline.py` script synthesizes Mealy machines from TLSF specifications, generates training data, and trains Moore SSM models. Point it at a directory of `.tlsf` files:

```bash
python drift_test_pipeline.py path/to/tlsf_dir/
```

To specify a custom output file (defaults to `training_results.json`):

```bash
python drift_test_pipeline.py path/to/tlsf_dir/ -o my_results.json
```
