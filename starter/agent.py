"""Required competition entry point.

The evaluator imports `Agent` from this exact module path
(`starter.agent.Agent`) and instantiates it as `Agent(catalog_path)`. The
real implementation -- Agent Shopper -- lives in the `agent_shopper`
package; this file is intentionally just a shim so that import path keeps
working unchanged while everything else stays testable in isolation.

See agent_shopper/agent.py and README.md for the architecture.
"""

from __future__ import annotations

from agent_shopper.agent import Agent

__all__ = ["Agent"]
