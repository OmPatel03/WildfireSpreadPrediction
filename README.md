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
