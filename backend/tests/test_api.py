"""
V&V – API / Backend Test Suite
================================
Covers the four areas requested for the submission document:

  1.  Endpoints tested          – GET /health, GET /catalog, POST /findSpread,
                                  POST /graphql (health / catalog / findSpread)
  2.  Example request/response  – asserted in every happy-path test
  3.  Health-check results      – TestHealth class
  4.  Invalid input / errors    – TestCatalogValidation, TestFindSpreadValidation,
                                  TestFindSpreadErrors, TestGraphQLErrors
  5.  Response-time budgets     – TestResponseTimes (< 1 s for mocked service)

Run with:
    cd backend
    pip install pytest httpx
    pytest tests/ -v
"""
from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from conftest import FAKE_GEOJSON, FAKE_PREDICTION, HEALTH_DICT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_GRAPHQL = "/graphql"
_GQL_CONTENT_TYPE = {"Content-Type": "application/json"}


def _gql(query: str, variables: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a GraphQL POST body."""
    payload: Dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    return payload


# ---------------------------------------------------------------------------
# 1. GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    """Health-check endpoint — GET /health"""

    def test_returns_200(self, client: TestClient) -> None:
        """Endpoint is reachable and reports HTTP 200."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_content_type_is_json(self, client: TestClient) -> None:
        r = client.get("/health")
        assert "application/json" in r.headers["content-type"]

    def test_response_body_schema(self, client: TestClient) -> None:
        """All required fields are present with correct types."""
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["device"] == "cpu"
        assert body["default_year"] == 2021
        assert isinstance(body["model_path"], str)
        assert isinstance(body["hdf5_root"], str)

    def test_example_response(self, client: TestClient) -> None:
        """
        Example response
        ----------------
        GET /health  →  200 OK
        {
            "status": "ok",
            "model_path": "/app/resources/model.ckpt",
            "model_loaded": true,
            "device": "cpu",
            "hdf5_root": "/data/HDF5",
            "default_year": 2021
        }
        """
        body = client.get("/health").json()
        assert body == HEALTH_DICT


# ---------------------------------------------------------------------------
# 2. GET /catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    """Catalog endpoint — GET /catalog"""

    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/catalog")
        assert r.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        body = client.get("/catalog").json()
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_item_has_required_fields(self, client: TestClient) -> None:
        """
        Example response item
        ---------------------
        GET /catalog  →  200 OK
        [
          {
            "fireId": "fire_00001",
            "year": 2021,
            "longitude": -120.457,
            "latitude": 38.892,
            "timeSteps": 10,
            "samples": 9,
            "height": 64,
            "width": 64,
            "hdf5Path": "/fake/2021/fire_00001.hdf5"
          }
        ]
        """
        item = client.get("/catalog").json()[0]
        assert item["fireId"] == "fire_00001"
        assert item["year"] == 2021
        assert abs(item["longitude"] - (-120.457)) < 0.001
        assert abs(item["latitude"] - 38.892) < 0.001
        assert item["samples"] == 9
        assert item["height"] == 64
        assert item["width"] == 64
        assert item["timeSteps"] == 10

    def test_year_query_param(self, client: TestClient) -> None:
        r = client.get("/catalog?year=2021")
        assert r.status_code == 200

    def test_limit_and_offset_params(self, client: TestClient) -> None:
        r = client.get("/catalog?limit=10&offset=0")
        assert r.status_code == 200


class TestCatalogValidation:
    """Catalog endpoint — invalid query-parameter handling (expect HTTP 422)."""

    def test_limit_zero_rejected(self, client: TestClient) -> None:
        """limit must be >= 1."""
        r = client.get("/catalog?limit=0")
        assert r.status_code == 422

    def test_limit_above_max_rejected(self, client: TestClient) -> None:
        """limit must be <= 500."""
        r = client.get("/catalog?limit=501")
        assert r.status_code == 422

    def test_negative_offset_rejected(self, client: TestClient) -> None:
        """offset must be >= 0."""
        r = client.get("/catalog?offset=-1")
        assert r.status_code == 422

    def test_non_integer_year_rejected(self, client: TestClient) -> None:
        r = client.get("/catalog?year=abc")
        assert r.status_code == 422

    def test_422_body_contains_detail(self, client: TestClient) -> None:
        """FastAPI validation errors include a 'detail' key."""
        body = client.get("/catalog?limit=0").json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# 3. POST /findSpread
# ---------------------------------------------------------------------------

_VALID_SPREAD_BODY: Dict[str, Any] = {
    "fireId": "fire_00001",
    "year": 2021,
    "sampleOffset": -1,
    "probabilityThreshold": 0.5,
}


class TestFindSpread:
    """findSpread endpoint — POST /findSpread  (happy paths)"""

    def test_returns_200(self, client: TestClient) -> None:
        r = client.post("/findSpread", json=_VALID_SPREAD_BODY)
        assert r.status_code == 200

    def test_response_has_threshold(self, client: TestClient) -> None:
        body = client.post("/findSpread", json=_VALID_SPREAD_BODY).json()
        assert body["threshold"] == 0.5

    def test_response_has_geojson_feature_collection(
        self, client: TestClient
    ) -> None:
        """
        Example response (abridged)
        ---------------------------
        POST /findSpread  { "fireId": "fire_00001", "year": 2021 }  →  200 OK
        {
            "threshold": 0.5,
            "sampleIndex" | "sample_index": 8,
            "totalSamples" | "total_samples": 9,
            "fire": { ... },
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "fire_00001",
                        "geometry": { "type": "Point", "coordinates": [...] },
                        "properties": {
                            "summary": { "precision": ..., "recall": ..., "f1": ... },
                            "prediction": { "mask": [[...]], "probabilities": [[...]] },
                            "groundTruthMask": [[...]]
                        }
                    }
                ]
            }
        }
        """
        body = client.post("/findSpread", json=_VALID_SPREAD_BODY).json()
        gj = body["geojson"]
        assert gj["type"] == "FeatureCollection"
        features = gj["features"]
        assert len(features) == 1
        feat = features[0]
        assert feat["type"] == "Feature"
        assert feat["id"] == "fire_00001"
        assert feat["geometry"]["type"] == "Point"

    def test_response_geojson_properties_schema(
        self, client: TestClient
    ) -> None:
        props = (
            client.post("/findSpread", json=_VALID_SPREAD_BODY)
            .json()["geojson"]["features"][0]["properties"]
        )
        assert "summary" in props
        assert "prediction" in props
        assert "groundTruthMask" in props
        summary = props["summary"]
        for key in ("precision", "recall", "f1", "positivePixels"):
            assert key in summary, f"summary missing key '{key}'"

    def test_response_sample_and_total_fields(
        self, client: TestClient
    ) -> None:
        body = client.post("/findSpread", json=_VALID_SPREAD_BODY).json()
        assert body["sampleIndex"] == 8
        assert body["totalSamples"] == 9

    def test_default_threshold_omitted(self, client: TestClient) -> None:
        """Omitting probabilityThreshold is valid (service uses its default)."""
        body = {"fireId": "fire_00001", "year": 2021}
        r = client.post("/findSpread", json=body)
        assert r.status_code == 200

    def test_snake_case_field_name_also_accepted(
        self, client: TestClient
    ) -> None:
        """populate_by_name=True means fire_id works as well as fireId."""
        body = {"fire_id": "fire_00001", "year": 2021}
        r = client.post("/findSpread", json=body)
        assert r.status_code == 200


class TestFindSpreadValidation:
    """findSpread endpoint — invalid-input handling (expect HTTP 422)."""

    def test_missing_fire_id_rejected(self, client: TestClient) -> None:
        r = client.post("/findSpread", json={"year": 2021})
        assert r.status_code == 422

    def test_empty_body_rejected(self, client: TestClient) -> None:
        r = client.post("/findSpread", json={})
        assert r.status_code == 422

    def test_threshold_above_1_rejected(self, client: TestClient) -> None:
        body = {**_VALID_SPREAD_BODY, "probabilityThreshold": 1.1}
        r = client.post("/findSpread", json=body)
        assert r.status_code == 422

    def test_threshold_below_0_rejected(self, client: TestClient) -> None:
        body = {**_VALID_SPREAD_BODY, "probabilityThreshold": -0.01}
        r = client.post("/findSpread", json=body)
        assert r.status_code == 422

    def test_422_detail_present(self, client: TestClient) -> None:
        body = client.post("/findSpread", json={}).json()
        assert "detail" in body


class TestFindSpreadErrors:
    """findSpread endpoint — server-side error propagation."""

    def test_fire_not_found_returns_500(
        self, client: TestClient, mock_svc: MagicMock
    ) -> None:
        """
        When the repository raises FileNotFoundError (fire ID not in catalog)
        and no custom exception handler is registered, FastAPI returns HTTP 500.
        The original mock return value is restored after the test.
        """
        body = {**_VALID_SPREAD_BODY, "fireId": "fire_99999"}
        with patch.object(
            mock_svc,
            "find_spread",
            side_effect=FileNotFoundError("fire_99999 not in catalog"),
        ):
            r = client.post("/findSpread", json=body)
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# 4. POST /graphql
# ---------------------------------------------------------------------------

_HEALTH_GQL = """
query {
    health {
        status
        modelLoaded
        device
        defaultYear
        modelPath
        hdf5Root
    }
}
"""

_CATALOG_GQL = """
query {
    catalog(year: 2021, limit: 10, offset: 0) {
        fireId
        year
        longitude
        latitude
        samples
        height
        width
    }
}
"""

_FIND_SPREAD_GQL = """
query FindSpread($fireId: String!, $year: Int) {
    findSpread(fireId: $fireId, year: $year) {
        sampleIndex
        totalSamples
        threshold
        fire {
            fireId
            year
            longitude
            latitude
        }
        geojson
    }
}
"""


class TestGraphQL:
    """GraphQL gateway — POST /graphql (Strawberry)"""

    def test_graphql_health_returns_200(self, client: TestClient) -> None:
        r = client.post(
            _GRAPHQL, json=_gql(_HEALTH_GQL), headers=_GQL_CONTENT_TYPE
        )
        assert r.status_code == 200

    def test_graphql_health_data(self, client: TestClient) -> None:
        """
        Example GraphQL request/response
        ----------------------------------
        POST /graphql
        { "query": "{ health { status modelLoaded device defaultYear } }" }

        Response:
        {
            "data": {
                "health": {
                    "status": "ok",
                    "modelLoaded": true,
                    "device": "cpu",
                    "defaultYear": 2021
                }
            }
        }
        """
        r = client.post(
            _GRAPHQL, json=_gql(_HEALTH_GQL), headers=_GQL_CONTENT_TYPE
        )
        data = r.json()
        assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
        h = data["data"]["health"]
        assert h["status"] == "ok"
        assert h["modelLoaded"] is True
        assert h["device"] == "cpu"
        assert h["defaultYear"] == 2021

    def test_graphql_catalog_returns_200(self, client: TestClient) -> None:
        r = client.post(
            _GRAPHQL, json=_gql(_CATALOG_GQL), headers=_GQL_CONTENT_TYPE
        )
        assert r.status_code == 200

    def test_graphql_catalog_data(self, client: TestClient) -> None:
        """
        Example GraphQL catalog response item:
        { "fireId": "fire_00001", "year": 2021, "longitude": -120.457, ... }
        """
        r = client.post(
            _GRAPHQL, json=_gql(_CATALOG_GQL), headers=_GQL_CONTENT_TYPE
        )
        body = r.json()
        assert "errors" not in body, f"GraphQL errors: {body.get('errors')}"
        items = body["data"]["catalog"]
        assert isinstance(items, list) and len(items) >= 1
        item = items[0]
        assert item["fireId"] == "fire_00001"
        assert item["year"] == 2021
        assert abs(item["longitude"] - (-120.457)) < 0.001
        assert item["samples"] == 9

    def test_graphql_find_spread_returns_200(self, client: TestClient) -> None:
        r = client.post(
            _GRAPHQL,
            json=_gql(_FIND_SPREAD_GQL, {"fireId": "fire_00001", "year": 2021}),
            headers=_GQL_CONTENT_TYPE,
        )
        assert r.status_code == 200

    def test_graphql_find_spread_data(self, client: TestClient) -> None:
        """
        Example GraphQL findSpread response:
        {
            "data": {
                "findSpread": {
                    "sampleIndex": 8,
                    "totalSamples": 9,
                    "threshold": 0.5,
                    "fire": { "fireId": "fire_00001", "year": 2021 },
                    "geojson": { "type": "FeatureCollection", ... }
                }
            }
        }
        """
        r = client.post(
            _GRAPHQL,
            json=_gql(_FIND_SPREAD_GQL, {"fireId": "fire_00001", "year": 2021}),
            headers=_GQL_CONTENT_TYPE,
        )
        body = r.json()
        assert "errors" not in body, f"GraphQL errors: {body.get('errors')}"
        spread = body["data"]["findSpread"]
        assert spread["sampleIndex"] == 8
        assert spread["totalSamples"] == 9
        assert spread["threshold"] == pytest.approx(0.5)
        assert spread["fire"]["fireId"] == "fire_00001"
        assert spread["fire"]["year"] == 2021
        # geojson is a JSON scalar — Strawberry returns the dict as-is
        assert isinstance(spread["geojson"], dict)
        assert spread["geojson"]["type"] == "FeatureCollection"


class TestGraphQLErrors:
    """GraphQL error-handling — invalid queries and malformed payloads."""

    def test_unknown_field_returns_error(self, client: TestClient) -> None:
        """
        Querying a field that does not exist on the schema returns a GraphQL
        error object.  Strawberry may respond with HTTP 200 + errors array,
        or HTTP 400/422 depending on configuration.
        """
        r = client.post(
            _GRAPHQL,
            json=_gql("{ nonExistentField }"),
            headers=_GQL_CONTENT_TYPE,
        )
        assert r.status_code in (200, 400, 422)
        if r.status_code == 200:
            assert "errors" in r.json()

    def test_syntax_error_returns_error(self, client: TestClient) -> None:
        """A syntactically invalid GraphQL query must be rejected."""
        r = client.post(
            _GRAPHQL,
            json=_gql("{ health { "),         # unterminated selection set
            headers=_GQL_CONTENT_TYPE,
        )
        assert r.status_code in (200, 400, 422)
        if r.status_code == 200:
            assert "errors" in r.json()

    def test_malformed_json_rejected(self, client: TestClient) -> None:
        """Non-JSON body must be rejected."""
        r = client.post(
            _GRAPHQL,
            content=b"not-json-at-all",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422)

    def test_empty_query_string_rejected(self, client: TestClient) -> None:
        """An empty query string must produce either an error response or 422/400."""
        r = client.post(
            _GRAPHQL,
            json={"query": ""},
            headers=_GQL_CONTENT_TYPE,
        )
        assert r.status_code in (200, 400, 422)
        if r.status_code == 200:
            assert "errors" in r.json()


# ---------------------------------------------------------------------------
# 5. Response-time budgets (< 1 s for a fully mocked service)
# ---------------------------------------------------------------------------


class TestResponseTimes:
    """
    Each mocked endpoint must complete within 1 second.
    This budget covers network I/O overhead in the ASGI test transport.
    Production budgets may differ based on model inference time.
    """

    def _measure(self, fn, *args, **kwargs) -> float:
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        return time.perf_counter() - t0

    def test_health_under_1s(self, client: TestClient) -> None:
        elapsed = self._measure(client.get, "/health")
        assert elapsed < 1.0, f"GET /health: {elapsed:.3f}s > 1.0s"

    def test_catalog_under_1s(self, client: TestClient) -> None:
        elapsed = self._measure(client.get, "/catalog")
        assert elapsed < 1.0, f"GET /catalog: {elapsed:.3f}s > 1.0s"

    def test_find_spread_under_1s(self, client: TestClient) -> None:
        elapsed = self._measure(
            client.post, "/findSpread", json=_VALID_SPREAD_BODY
        )
        assert elapsed < 1.0, f"POST /findSpread: {elapsed:.3f}s > 1.0s"

    def test_graphql_health_under_1s(self, client: TestClient) -> None:
        elapsed = self._measure(
            client.post,
            _GRAPHQL,
            json=_gql("{ health { status } }"),
            headers=_GQL_CONTENT_TYPE,
        )
        assert elapsed < 1.0, f"POST /graphql (health): {elapsed:.3f}s > 1.0s"

    def test_graphql_catalog_under_1s(self, client: TestClient) -> None:
        elapsed = self._measure(
            client.post,
            _GRAPHQL,
            json=_gql(_CATALOG_GQL),
            headers=_GQL_CONTENT_TYPE,
        )
        assert elapsed < 1.0, f"POST /graphql (catalog): {elapsed:.3f}s > 1.0s"

    def test_graphql_find_spread_under_1s(self, client: TestClient) -> None:
        elapsed = self._measure(
            client.post,
            _GRAPHQL,
            json=_gql(
                _FIND_SPREAD_GQL, {"fireId": "fire_00001", "year": 2021}
            ),
            headers=_GQL_CONTENT_TYPE,
        )
        assert elapsed < 1.0, f"POST /graphql (findSpread): {elapsed:.3f}s > 1.0s"
