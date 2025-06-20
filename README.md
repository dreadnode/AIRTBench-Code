# AIRTBench: Autonomous AI Red Teaming Agent Code

<div align="center">

<img
  src="https://d1lppblt9t2x15.cloudfront.net/logos/5714928f3cdc09503751580cffbe8d02.png"
  alt="Logo"
  align="center"
  width="144px"
  height="144px"
/>

</div>

<!-- BEGIN_AUTO_BADGES -->
<div align="center">

[![Pre-Commit](https://github.com/dreadnode/AIRTBench-Code/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/dreadnode/AIRTBench-Code/actions/workflows/pre-commit.yaml)
[![Renovate](https://github.com/dreadnode/AIRTBench-Code/actions/workflows/renovate.yaml/badge.svg)](https://github.com/dreadnode/AIRTBench-Code/actions/workflows/renovate.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/dreadnode/AIRTBench-Code)](https://github.com/dreadnode/AIRTBench-Code/releases)

[![arXiv](https://img.shields.io/badge/arXiv-AIRTBench-b31b1b.svg)](https://arxiv.org/abs/2506.14682)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-ffca28.svg)](https://huggingface.co/datasets/dreadnode/AIRTBench/blob/main/README.md)
[![Dreadnode](https://img.shields.io/badge/Dreadnode-Blog-5714928f.svg)](https://dreadnode.io/blog/ai-red-team-benchmark)
[![Agent Harness](https://img.shields.io/badge/📚_Agent_Harness-Documentation-5714928f.svg)](https://docs.dreadnode.io/strikes/how-to/airtbench-agent)

[![GitHub stars](https://img.shields.io/github/stars/dreadnode/AIRTBench-Code?style=social)](https://github.com/dreadnode/AIRTBench-Code/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/dreadnode/AIRTBench-Code/pulls)

</div>
<!-- END_AUTO_BADGES -->

---

This repository contains the implementation of the AIRTBench autonomous AI red teaming agent, complementing our research paper [AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models](https://arxiv.org/abs/2506.14682) and accompanying blog post, "[Do LLM Agents Have AI Red Team Capabilities? We Built a Benchmark to Find Out](https://dreadnode.io/blog/ai-red-team-benchmark)".

The AIRTBench agent is designed to evaluate the autonomous red teaming capabilities of large language models (LLMs) through AI/ML Capture The Flag (CTF) challenges. Our agent systematically exploits LLM-based targets by solving challenges on the Dreadnode Strikes platform, providing a standardized benchmark for measuring adversarial AI capabilities.

- [AIRTBench: Autonomous AI Red Teaming Agent Code](#airtbench-autonomous-ai-red-teaming-agent-code)
  - [Agent Harness Construction](#agent-harness-construction)
  - [Setup](#setup)
  - [Documentation](#documentation)
  - [Run the Evaluation](#run-the-evaluation)
    - [Basic Usage](#basic-usage)
    - [Challenge Filtering](#challenge-filtering)
  - [Resources](#resources)
  - [Dataset](#dataset)
  - [Citation](#citation)
  - [Model requests](#model-requests)
  - [🤝 Contributing](#-contributing)
  - [🔐 Security](#-security)
  - [⭐ Star History](#-star-history)

## Agent Harness Construction

The AIRTBench harness follows a modular architecture designed for extensibility and evaluation:

<div align="center">
  <img src="assets/airtbench_architecture_diagram_dark.png" alt="AIRTBench Architecture" width="100%">
  <br>
  <em>Figure: AIRTBench harness construction architecture showing the interaction between agent components, challenge interface, and evaluation framework.</em>
</div>

## Setup

You can setup the virtual environment with uv:

```bash
uv sync
```

## Documentation

Technical documentation for the AIRTBench agent is available in the [Dreadnode Strikes documentation](https://docs.dreadnode.io/strikes/how-to/airtbench-agent).

## Run the Evaluation

<mark>In order to run the code, you will need access to the Dreadnode strikes platform, see the [docs](https://docs.Dreadnode.io/strikes/overview) or submit for the Strikes waitlist [here](https://platform.dreadnode.io/waitlist/strikes)</mark>.

This [rigging](https://docs.dreadnode.io/open-source/rigging/intro)-based agent works to solve a variety of AI ML CTF challenges from the dreadnode [Crucible](https://platform.dreadnode.io/crucible) platform and given access to execute python commands on a network-local container with custom [Dockerfile](./airtbench/container/Dockerfile).

```bash
uv run -m airtbench --help
```

### Basic Usage

```bash
uv run -m airtbench --model $MODEL --project $PROJECT --platform-api-key $DREADNODE_TOKEN --token $DREADNODE_TOKEN --server https://platform.dreadnode.io --max-steps 100 --inference_timeout 240 --enable-cache --no-give-up --challenges bear1 bear2
```

### Challenge Filtering

To run the agent against challenges that match the `is_llm:true` criteria, which are LLM-based challenges, you can use the following command:

```bash
uv run -m airtbench --model <model> --llm-challenges-only
```

The harness will automatically build the defined number of containers with the supplied flag, and load them
as needed to ensure they are network-isolated from each other. The process is generally:

1. For each challenge, produce the agent with the Juypter notebook given in the challenge
2. Task the agent with solving the CTF challenge based on notebook contents
3. Bring up the associated environment
4. Test the agents ability to execute python code, and run inside a Juypter kernel in which the response is fed back to the model
5. If the CTF challenge is solved and flag is observed, the agent must submit the flag
6. Otherwise run until an error, give up, or max-steps is reached

Check out [the challenge manifest](./airtbench/challenges/.challenges.yaml) to see current challenges in scope.

## Resources

- [📄 Paper on arXiv](https://arxiv.org/abs/2506.14682)
- [📝 Blog post](https://dreadnode.io/blog/ai-red-team-benchmark)

## Dataset

- Download the dataset directly from [🤗Hugging Face](https://huggingface.co/datasets/dreadnode/AIRTBench/blob/main/README.md)
- Instructions for loading the dataset can be found in the [dataset](./dataset/README.md) directory also.

## Citation

If you find our work helpful, please use the following citations.

```bibtex
@misc{dawson2025airtbenchmeasuringautonomousai,
      title={AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models},
      author={Ads Dawson and Rob Mulla and Nick Landers and Shane Caldwell},
      year={2025},
      eprint={2506.14682},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.14682},
}
```

## Model requests

If you know of a model that may be interesting to analyze, but do not have the resources to run it yourself, feel free to open a feature request via a GitHub issue.

## 🤝 Contributing

Forks and contributions are welcome! Please see our [Contributing Guide](docs/contributing.md).

## 🔐 Security

See our [Security Policy](SECURITY.md) for reporting vulnerabilities.

## ⭐ Star History

[![GitHub stars](https://img.shields.io/github/stars/dreadnode/AIRTBench-Code?style=social)](https://github.com/dreadnode/AIRTBench-Code/stargazers)

By watching the repo, you can also be notified of any upcoming releases.

[![Star history graph](https://api.star-history.com/svg?repos=dreadnode/AIRTBench-Code&type=Date)](https://star-history.com/#dreadnode/AIRTBench-Code&Date)
