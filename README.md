# Easy Inspect

A high-level, zero-code interface for evaluating LLMs built on top of [Inspect-AI](https://inspect.ai-safety-institute.org.uk/).

## Overview

Easy Inspect provides a simple way to evaluate language models using YAML configuration files. It handles all the complexity of setting up evaluation tasks, running models, and analyzing results.

## Features

- Define evaluation questions using simple YAML files
- Support for multiple question types:
  - Free-form text responses
  - Numerical ratings (0-100)
  - Model-graded evaluations
- Built-in support for multiple LLM providers (OpenAI, Anthropic)
- Automatic result caching and logging
- Easy results analysis and visualization

## Installation

```bash
git clone https://github.com/dtch1997/easy-eval.git
cd easy-eval
pip install -e .
```

## Usage

See the [examples](examples) directory for usage examples.
