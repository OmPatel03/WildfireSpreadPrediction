# WildfireSpreadPrediction

## Backend API (FastAPI + GraphQL)

The backend lives under `backend/src` and exposes both REST and GraphQL interfaces for the wildfire spread model checkpoint in `backend/resources/model.ckpt`. It expects the pre-generated HDF5 cubes under `../HDF5/<year>` (relative to the repository root), which matches the layout you shared.

### Prerequisites

1. Activate the provided conda environment when running commands:
   ```powershell
   conda run -n wildfire python --version
   ```
2. Ensure the following Python packages are available in the `wildfire` environment: `fastapi`, `strawberry-graphql`, `uvicorn[standard]`, `torch`, `torchvision`, `segmentation-models-pytorch`, `pytorch-lightning`, `h5py`.

### Running the service locally

From the repository root:

```powershell
conda run -n wildfire uvicorn main:app --app-dir backend/src --reload --host 0.0.0.0 --port 8000
```

The app loads the PyTorch Lightning checkpoint once at startup, scans the HDF5 directory for available fires, and serves both REST (`/health`, `/catalog`, `/findSpread`) and GraphQL (`/graphql`) endpoints.

### Configuration

Environment variables (optional overrides):

| Variable | Default | Description |
| --- | --- | --- |
| `HDF5_ROOT` | `../HDF5` (relative to repo root) | Absolute or relative path to the folder that contains yearly subfolders with `.hdf5` cubes. |
| `MODEL_CHECKPOINT` | `backend/resources/model.ckpt` | Path to the Lightning checkpoint to load. |
| `WSTS_ROOT` | `../../../src/wsts/` (relative to directory of `wsts_bridge.py`) | Path to WildfireSpreadTS (`wsts`) source code directory root. |
| `WILDFIRE_DEFAULT_YEAR` | `2021` | Year used when the client does not specify one. |
| `WILDFIRE_STATS_YEARS` | `2018,2019` | Training years whose statistics are used for feature standardization. |
| `WILDFIRE_N_LEADS` | `1` | Number of leading observations to feed into the model per prediction. |
| `WILDFIRE_PROB_THRESHOLD` | `0.5` | Default probability cutoff when converting model logits to a binary mask. |
| `WILDFIRE_DEVICE` | auto-detect | Set to `cpu`, `cuda`, etc. to pin inference to a specific device. |

### REST endpoints

* `GET /health` – returns model/device status plus resolved paths.
* `GET /catalog?year=2021&limit=50&offset=0` – lists available fires (id, extent, number of samples) for the requested year.
* `POST /findSpread` – body:
  ```json
  {
    "fireId": "fire_24935867",
    "year": 2021,
    "sampleOffset": -1,
    "probabilityThreshold": 0.5
  }
  ```
  Responds with metadata plus a GeoJSON FeatureCollection embedding:
  * Centroid coordinates from the HDF5 attributes (`lnglat`) so the client can place the feature on a basemap.
  * The predicted probability raster and binary mask for the requested sample window.
  * Basic summary stats (mean/max probabilities, positive pixel counts, etc.).

### GraphQL endpoint

Available at `/graphql`. Example query:

```graphql
query {
  catalog(year: 2021, limit: 5) {
    fireId
    year
    samples
  }
  findSpread(fireId: "fire_24935867") {
    fire {
      fireId
      latitude
      longitude
    }
    threshold
    geojson
  }
}
```

The GraphQL types mirror the REST responses and surface the same GeoJSON payload for easy client reuse.


## Containerization
Components of the `WildfireSpreadPrediction` application are containerized for ease of setup and execution.

Assume the below commands should be run from the repository root (`WildfireSpreadPrediction/`). If using a different
container runtime/CLI, substitute `podman` with this (e.g. `docker`).

### Building Images

#### Build Backend Base Image
The backend base image packages Python and most libraries for convenience. Separation of the backend base image
from the final, runnable image allows for quicker builds, assuming there are no major changes of requirements.

To build the backend base image, run:

```bash
podman build --network host -f backend/backend.base.Containerfile -t wispr-backend-base .
```
- In RHEL/SELinux enabled operating systems, builds may fail without additional configuration.
For ease of execution, this can be run as root with `sudo`
(note that the built image will then only be accessible to root)
- The `--network host` flag may not be needed depending on network settings,
but helps avoid e.g. DNS issues for installing dependencies within container

#### Build Backend Image
Once the backend base image is built or pulled, it can be used as the base image to build the final backend image.
The build process for this image should just be to copy the source code into the container image, but if additional
or modified requirements are needed, these can also be installed at this stage for development/debugging purposes.
However, it is recommended that the final changes in requirements be eventually migrated in the base image, too,
in order to minimize build time for the final image.

To build the backend image, run:

```bash
podman build -f backend/backend.Containerfile -t wispr-backend .
```
- The same notes as above apply, especially if installing additional requirements

## Running Containers
For convenience, it is easiest to run the application containers using `podman compose`.  

The `compose.yaml` file defines several services, the main ones being `frontend` and `backend`.
For each service, the following may be defined:
- Which Containerfile is used to build the container image
- What command to execute when running the container
- What environment variables to set in the container
- Which ports to map from the host to the container
- Which volumes to mount into the container

... and so on. This avoids having to manually set these as flags in regular `podman` commands (e.g. `build`, `run`).

To run the application, the simplest option is to run:

```bash
podman compose up -d
```

Then, to stop and remove the container(s), run:
```bash
podman compose down
```