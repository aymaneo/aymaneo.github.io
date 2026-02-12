---
title: "Why Rust for ML Infrastructure"
description: "Making the case for Rust in ML infrastructure — performance, safety, and the surprisingly great ecosystem for building data-intensive systems."
date: 2025-02-20
tags: ["Rust", "ML", "Infrastructure", "Performance"]
---

Python dominates ML research. That's fine. But when it comes to the infrastructure *around* ML — the serving layers, data pipelines, and orchestration systems — there's a strong case for Rust.

## The Performance Argument

This isn't just about being fast. It's about being *predictably* fast. ML serving has strict latency requirements, and Python's GIL and garbage collector make tail latencies unpredictable.

```rust
use tokio::time::Instant;

async fn serve_prediction(
    model: &ModelHandle,
    input: &Tensor,
) -> Result<Prediction, ServeError> {
    let start = Instant::now();

    let preprocessed = preprocess(input)?;
    let output = model.forward(&preprocessed).await?;
    let prediction = postprocess(&output)?;

    metrics::histogram!(
        "inference_latency_ms",
        start.elapsed().as_millis() as f64
    );

    Ok(prediction)
}
```

With Rust, p99 latencies are within 2x of p50. In Python, that ratio can be 10x or worse.

## Safety in Data Pipelines

ML data pipelines are notoriously fragile. Silent data corruption is the enemy. Rust's type system catches entire categories of bugs at compile time:

- No null pointer exceptions in your feature extraction
- No accidental type coercion in your data transforms
- No race conditions in your parallel preprocessing

## The Ecosystem is Ready

The Rust ML ecosystem has matured significantly:

- **`candle`** — Minimalist ML framework from Hugging Face
- **`tokenizers`** — The tokenizer library everyone uses (it's Rust under the hood)
- **`lance`** — Columnar data format optimized for ML
- **`qdrant`** — Vector similarity search engine

You don't need to rewrite your training loops in Rust. But the infrastructure that feeds data in and serves predictions out? That's where Rust shines.

## Practical Advice

Start small. Pick one component — maybe a preprocessing service or a custom data loader — and build it in Rust. Measure the difference. The results usually speak for themselves.
