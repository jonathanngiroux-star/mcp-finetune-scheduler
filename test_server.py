import pytest
import asyncio
import json
from mcp_finetune_scheduler.server import (
    schedule_finetune, 
    run_regression_eval, 
    compare_model_outputs,
    register_model,
    list_jobs,
    list_models,
    list_tools_handler,
    call_tool_handler
)

@pytest.fixture
async def clean_db():
    import sqlite3
    from mcp_finetune_scheduler.server import DB_PATH
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM finetune_jobs")
        conn.execute("DELETE FROM model_registry")
    yield
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM finetune_jobs")
        conn.execute("DELETE FROM model_registry")

@pytest.mark.asyncio
async def test_list_tools():
    from mcp_finetune_scheduler.server import list_tools_handler
    from mcp.types import ListToolsResult
    result = await list_tools_handler()
    assert isinstance(result, ListToolsResult)
    assert len(result.tools) == 6

@pytest.mark.asyncio
async def test_schedule_finetune(clean_db):
    from mcp_finetune_scheduler.server import schedule_finetune
    import secrets
    unique_name = f"Test Job {secrets.token_hex(4)}"
    job = await schedule_finetune({
        "name": unique_name,
        "base_model": "meta-llama/Llama-2-7b-hf",
        "dataset_id": "dataset-1",
        "hyperparameters": {"lr": 2e-4, "epochs": 3}
    })
    assert len(job) > 0
    data = json.loads(job[0].text)
    assert data["status"] == "pending"
    assert "job_id" in data

@pytest.mark.asyncio
async def test_run_regression_eval(clean_db):
    from mcp_finetune_scheduler.server import run_regression_eval
    result = await run_regression_eval({"model_path": "./models/test", "eval_dataset": "./eval"})
    assert len(result) > 0
    data = json.loads(result[0].text)
    assert "perplexity" in data
    assert "accuracy" in data

@pytest.mark.asyncio
async def test_compare_model_outputs(clean_db):
    from mcp_finetune_scheduler.server import compare_model_outputs
    result = await compare_model_outputs({
        "model1_path": "./models/model1",
        "model2_path": "./models/model2",
        "eval_dataset": "./eval"
    })
    data = json.loads(result[0].text)
    assert "similarity" in data
    assert "diff_count" in data

@pytest.mark.asyncio
async def test_register_model(clean_db):
    from mcp_finetune_scheduler.server import register_model
    model = await register_model({
        "name": "Test Model",
        "base_model": "llama-2-7b",
        "version": "1.0",
        "path": "./models/test",
        "metrics": {"perplexity": 12.5, "accuracy": 0.87}
    })
    data = json.loads(model[0].text)
    assert data["status"] == "registered"
    assert data["model_id"] == "Test Model-v1.0"

@pytest.mark.asyncio
async def test_list_jobs(clean_db):
    from mcp_finetune_scheduler.server import schedule_finetune, list_jobs
    import secrets
    unique_name = f"Test Job {secrets.token_hex(4)}"
    await schedule_finetune({
        "name": unique_name,
        "base_model": "llama-2-7b",
        "dataset_id": "ds-1",
        "hyperparameters": {"lr": 2e-4, "epochs": 3}
    })
    jobs = await list_jobs({})
    data = json.loads(jobs[0].text)
    assert len(data["jobs"]) >= 1

@pytest.mark.asyncio
async def test_list_models(clean_db):
    from mcp_finetune_scheduler.server import register_model, list_models
    await register_model({
        "name": "Model 1", "base_model": "llama-2-7b", "version": "1.0",
        "path": "./m1", "metrics": {}
    })
    await register_model({
        "name": "Model 2", "base_model": "llama-2-13b", "version": "1.0",
        "path": "./m2", "metrics": {}
    })
    models = await list_models({})
    data = json.loads(models[0].text)
    assert len(data["models"]) >= 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])