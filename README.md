# Code for the "AIRTBench" AI Red Teaming Agent

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

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Super-Linter](https://github.com/dreadnode/AIRTBench/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![Spell-Check](https://github.com/dreadnode/AIRTBench/actions/workflows/spellcheck.yml/badge.svg)](https://github.com/rojopolis/spellcheck-github-actions)

</div>
<!-- END_AUTO_BADGES -->

---

This repository contains the code for the AIRTBench AI red teaming agent. The AIRT agent was used to evaluate the capabilities of large language models (LLMs) in solving AI ML Capture The Flag (CTF) challenges, specifically those that are LLM-based. The agent is designed to autonomously exploit LLMs by solving challenges on the Dreadnode Strikes platform.

The paper is available on [arXiV](TODO) and [ACL Anthology](TODO).

- [Code for the "AIRTBench" AI Red Teaming Agent](#code-for-the-airtbench-ai-red-teaming-agent)
  - [Setup](#setup)
  - [Run the Evaluation](#run-the-evaluation)
  - [Model requests](#model-requests)

## Setup

You can setup the virtual environment with uv:

```bash
uv sync
```

## Run the Evaluation

<mark>In order to run the code, you will need access to the Dreadnode strikes platform, see the [docs](https://docs.Dreadnode.io/strikes/overview) or submit for the Strikes waitlist [here](https://platform.dreadnode.io/waitlist/strikes)</mark>.

This [rigging](https://docs.dreadnode.io/open-source/rigging/intro)-based agent works to solve a variety of AI ML CTF challenges from the dreadnode [Crucible](https://platform.dreadnode.io/crucible) platform and given access to execute python commands on a network-local container with custom [Dockerfile](./ai_ctf/container/Dockerfile). This example-agent is also a compliment to our research paper [AIRTBench: Can Language Models Autonomously Exploit
Language Models?](https://arxiv.org/abs/TODO). # TODO: Add link to paper once published.

```bash
uv run -m ai_ctf --help
```

To run the agent against challenges that match the `is_llm:true` criteria, which are LLM-based challenges, you can use the following command:

```bash
uv run -m ai_ctf --model <model> --llm-challenges-only
```

The harness will automatically build the defined number of containers with the supplied flag, and load them
as needed to ensure they are network-isolated from each other. The process is generally:

1. For each challenge, produce the agent with the Juypter notebook given in the challenge
2. Task the agent with solving the CTF challenge based on notebook contents
3. Bring up the associated environment
4. Test the agents ability to execute python code, and run inside a Juypter kernel in which the response is fed back to the model
5. If the CTF challenge is solved and flag is observed, the agent must submit the flag
6. Otherwise run until an error, give up, or max-steps is reached

Check out [the challenge manifest](./ai_ctf/challenges/.challenges.yaml) to see current challenges in scope.


## Model requests

If you know of a model that may be interesting to analyze, but do not have the resources to run it yourself, feel free to open a feature request via a GitHub issue.