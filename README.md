# mcp-finetune-scheduler

MCP server for fine-tuning subscription - schedule fine-tune runs, run regression evaluations, compare model outputs.

## Tools

### schedule_finetune_run
Schedule a fine-tuning run.

**Parameters:**
- `model_name` (string, required): Base model to fine-tune
- `dataset` (string, required): Dataset identifier
- `hyperparameters` (object, required): Training hyperparameters
- `schedule` (string, optional): "immediate", "scheduled", or "recurring" (default: "immediate")
- `schedule_time` (string, optional): ISO timestamp for scheduled runs (required if schedule is "scheduled")
- `recurrence` (string, optional): Cron expression for recurring runs

### run_regression_eval
Run regression evaluation on a fine-tuned model.

**Parameters:**
- `model_id` (string, required): Model identifier
- `benchmark_dataset` (string, required): Benchmark dataset to evaluate on
- `metrics` (array, optional): Metrics to compute - perplexity, accuracy, f1, bleu, rouge (default: ["perplexity", "accuracy"])

### compare_model_outputs
Compare outputs from multiple models on the same prompts.

**Parameters:**
- `model_ids` (array, required): Model IDs to compare
- `prompts` (array, required): Test prompts
- `metrics` (array, optional): Comparison metrics - similarity, diversity, quality, latency (default: ["similarity", "quality"])

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Connecting to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "finetune-scheduler": {
      "command": "python",
      "args": ["/path/to/mcp-finetune-scheduler/main.py"],
      "cwd": "/path/to/mcp-finetune-scheduler"
    }
  }
}
```

## License

MIT