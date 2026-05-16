---
title: "Inside async RL training: generation efficiency and weight synchronization"
description: "A technical look at the two core problems in async RL training: the generation bottleneck and the weight synchronization challenge it creates."
date: 2026-05-16
tags: ["RL training", "LLM post-training", "infrastructure", "async RL"]
draft: false
---

Over the past few months I have been running experiments with async RL frameworks for LLM post-training, and spent part of that time debugging things that were slower than they should have been. In the process I ended up digging deeper into the underlying mechanisms than I initially expected.

This post is an attempt to share some of what I found, focusing on two problems: the generation bottleneck that motivates async RL, and the weight synchronization challenge it creates.

## The generation bottleneck

Running RL training on a language model synchronously looks straightforward on paper: generate rollouts, score them, train on them, repeat. In practice, the generation step dominates everything else, and by a significant margin.

The reason is structural. Training is compute-bound: you do a forward and backward pass over a large batch, and modern GPUs handle this well. Generation is slow for a fundamental reason: autoregressive decoding is inherently sequential. Each token depends on all previous ones, so you cannot parallelize within a sequence. Every token requires its own forward pass through the full model. With sequences of 8k, 16k, or 32k tokens, that compounds quickly. In representative RL training profiles on reasoning tasks, the rollout phase accounts for [up to 70% of total wall-clock time](https://arxiv.org/abs/2603.23414), with individual completions regularly reaching 16k tokens or more.

The ratio gets worse as tasks get harder. Short-answer tasks have short rollouts. Reasoning tasks don't, since long chains of thought are the point. The longer the expected rollout, the worse the synchronous RL bottleneck becomes.

GRPO makes this more acute. Where PPO generates one completion per prompt, GRPO samples a group, typically 8 to 16 completions, and computes advantages relative to the group mean. Every completion in a group must finish before training can begin. If one completion takes three times longer than average, the entire group waits. This is the straggler problem: tail latency determines effective batch time, not average latency.

The structural fix is to decouple inference from training entirely. Instead of running them in lockstep, you run them in parallel: a dedicated inference engine generates rollouts continuously and feeds a queue, while the trainer consumes from that queue independently. Each runs at its own pace. The training GPU stops waiting.

This is async RL, and it works. Moving to an asynchronous setup, [Ai2 achieved a 4x throughput improvement](https://finbarr.ca/making-rl-fast/) on Olmo 3's post-training through a combination of optimizations that the async architecture made possible: continuous batching, better actor threading, and inflight weight updates.

But decoupling creates a new problem immediately. If inference and training run as separate processes, updated model weights need to move from the trainer to the inference engine after each training step. How you do that, the mechanism, the timing, the cost, turns out to matter a lot for the efficiency you actually get. That is what the next section is about.

## Weight synchronization: the new bottleneck

Decoupling inference from training solves the idle GPU problem, but creates an immediate question: when the trainer updates the model weights, how do those updated weights reach the inference engine? In a synchronous setup this is trivial, both processes share the same weights. In an async setup, they are separate processes, often on separate GPUs or separate machines entirely. Getting weights from one to the other reliably and quickly is the weight sync problem, and it is less straightforward than it looks.

There are four main approaches in use today, each making a different set of tradeoffs.

**Filesystem broadcast** is the simplest. After each training step, the trainer writes updated weights to a shared filesystem. The inference engine reads them from there. No special networking required, works across any cluster. The cost is I/O bandwidth. For a 10B parameter model in BF16, that is roughly 20GB to write and read on every sync. On a shared NFS filesystem typical of HPC clusters, this takes minutes. At larger scales it becomes worse: Ant Group, in their [Awex paper](https://medium.com/@shawn.ck.yang/awex-an-ultra-fast-weight-sync-framework-powering-trillion-scale-reinforcement-learning-766ebc79f58b), notes that a trillion-parameter model would take hours via shared filesystem.

In practice, filesystem broadcast can fail in non-obvious ways. In my own runs, checkpoint saves to the same NFS mount, typically 20-30GB for model weights and optimizer state, would saturate available I/O bandwidth at the exact moment the inference engine was trying to read updated weights. The result was the inference server stalling for hours, producing no rollouts, while the trainer sat idle waiting for data that never arrived.

**NCCL broadcast** uses GPU-to-GPU collective communication, the same mechanism used for gradient synchronization during training. It bypasses the filesystem entirely and operates at GPU memory bandwidth. The catch is that NCCL collectives require all participating processes to share a process group, which means the trainer and inference engine need to be initialized together and remain coupled at the communication level. This constrains how you deploy and isolate the two components, and not all frameworks support it cleanly for every configuration.

**CUDA IPC** maps GPU memory regions directly between processes on the same node using CUDA inter-process communication. This eliminates serialization and network overhead entirely. [Biao He documented reducing weight sync time from 60 seconds to 7 seconds](https://hebiao064.github.io/rl-weight-sync) on a single-node setup by switching to CUDA IPC. The limitation is hard: it only works when trainer and inference share the same physical machine.

**RDMA** is what large-scale production systems use when they need weight sync to be fast across multiple nodes. Rather than routing data through the CPU or a filesystem, RDMA writes GPU memory contents directly to the memory of a remote machine over a high-speed fabric. [Perplexity Research achieved 1.3-second weight transfer](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds) for Kimi-K2, a trillion-parameter model, across 256 training GPUs to 128 inference GPUs. RDMA requires dedicated high-speed networking such as InfiniBand or RoCE.

In most cases you do not freely choose from all four options. You inherit a default from your framework and work within it, which makes understanding the tradeoffs important even when you cannot directly control them.

Sync strategy is the final dimension, often overlooked. The naive approach is to wait for all in-flight generations to finish before syncing weights, then restart inference. This drains the queue on every sync and reintroduces exactly the idle time async RL was designed to eliminate. A better approach, which Ai2 used in Olmo 3, is to update weights while inference continues running: ongoing generations complete with the previous weights, new ones immediately pick up the updated weights. The queue never empties. This single change accounted for a [117% throughput improvement](https://finbarr.ca/making-rl-fast/) in their reported results.

---

Async RL is a moving target. The approaches described here represent the current state, but the tooling, framework support, and understanding of staleness tolerance are all still evolving.
