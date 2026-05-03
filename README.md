# mlops-churn-platform

A model-agnostic MLOps platform demonstrating the end-to-end ML lifecycle on customer
churn (Telco dataset). The platform serves any of three swappable models — Logistic
Regression, XGBoost, or a PyTorch tabular MLP — behind a single registry contract.

> **Status:** under construction. Building module-by-module — see commit history.

## Why this exists

This repo backs résumé claims about reproducible training pipelines, experiment
tracking, model registries, online serving, drift monitoring, and **model-agnostic**
architectures. The point isn't "I trained a churn model" — it's "I built a platform
that can serve whichever of three models is currently champion, with full
reproducibility and drift surveillance."

## Quick start (preview — full flow lands as modules complete)

```bash
git clone https://github.com/<your-handle>/mlops-churn-platform.git
cd mlops-churn-platform
make install-dev
make test
```

## Roadmap

See commit history for the building order. Sections expand as modules ship.
