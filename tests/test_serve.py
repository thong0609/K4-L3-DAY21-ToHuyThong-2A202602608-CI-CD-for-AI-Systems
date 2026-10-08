from unittest.mock import Mock

import joblib
import pytest
from fastapi.testclient import TestClient

from src import serve


@pytest.fixture
def client(monkeypatch):
    model = Mock()
    model.predict.return_value = [1]
    monkeypatch.setattr(serve, "download_model", lambda: model)
    with TestClient(serve.app) as client:
        yield client, model


def test_score_schema_and_feature_order(client):
    api, model = client
    assert api.get("/healthz").json() == {"status": "ok"}
    values = [28, 2, 14, 2, 11, 0, 1, 0, 0, 45]
    response = api.post("/score", json={"features": values})
    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "label": "thu_nhap_cao"}
    row = model.predict.call_args.args[0]
    assert list(row.columns) == serve.FEATURE_NAMES
    assert row.iloc[0].tolist() == values
    model.predict.return_value = [0]
    assert api.post("/score", json={"features": values}).json()["label"] == "thu_nhap_thap"


def test_reject_wrong_feature_count(client):
    api, model = client
    assert api.post("/score", json={"features": [1, 2]}).status_code == 400
    model.predict.assert_not_called()


def test_download_from_expected_s3_location(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    s3 = Mock()
    s3.download_file.side_effect = lambda bucket, key, path: joblib.dump({"model": "test"}, path)
    monkeypatch.setattr(serve.boto3, "client", lambda *args, **kwargs: s3)
    assert serve.download_model() == {"model": "test"}
    assert s3.download_file.call_args.args[:2] == ("test-bucket", "artifacts/current/model.joblib")
    assert (tmp_path / "model.joblib").exists()


def test_startup_fails_when_download_fails(monkeypatch):
    monkeypatch.setattr(serve, "download_model", Mock(side_effect=RuntimeError("S3 unavailable")))
    with pytest.raises(RuntimeError, match="S3 unavailable"):
        with TestClient(serve.app):
            pass
