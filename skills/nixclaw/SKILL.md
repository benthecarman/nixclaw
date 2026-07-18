---
name: nixclaw
description: Safely inspect and improve a NixOS inference environment through the constrained NixClaw broker.
version: 0.1.0
platforms: [linux]
metadata:
  hermes:
    tags: [nixos, vllm, optimization, safety]
    category: devops
    requires_toolsets: [terminal]
---

# NixClaw

## When to use

Use this skill when asked to inspect, optimize, or explain the managed NixOS and
vLLM environment. NixClaw improves the environment through measured,
declarative experiments. It does not grant authority to change the live host.

## Safety boundary

- Use only `nixclaw-agent` for broker and lesson access.
- Never attempt `nixos-rebuild`, SSH, Docker, the Nix daemon, or the activator.
- Never ask to disable approval, health checks, the rollback lease, or
  containment.
- Treat broker rejection as final. Explain it; do not seek a bypass.
- Automatic optimization is limited to broker-advertised vLLM profile fields.
- Package, service, model, parser, arbitrary argument, listener, firewall, and
  free-form Nix changes require a separately reviewed proposal.

These instructions describe the workflow but are not the security boundary.
OpenShell and the broker enforce the actual authority.

## Procedure

1. Run `nixclaw-agent facts` and `nixclaw-agent config`.
2. Run `nixclaw-agent lessons search --workload agent-tools` before proposing a
   change. Exact lessons are directly relevant; transfer lessons require a new
   experiment.
3. State one measurable hypothesis and preserve the current generation as the
   base.
4. Run `nixclaw-agent optimize --workload agent-tools`. The optimizer selects
   one legal, untried patch and submits it idempotently.
5. Report the experiment ID and stop at `awaitingApproval`. Only the host
   operator can review or activate it.
6. After the operator runs the experiment, use
   `nixclaw-agent experiments sync ID` to ingest its current state.
7. For accepted experiments, report the validated lesson and measured gates.
   For rejected or rolled-back experiments, report the failure as negative
   evidence and do not retry the same patch.

## Verification

- `nixclaw-agent experiments show ID` matches the broker record.
- A terminal accepted experiment appears in a fresh
  `nixclaw-agent lessons search` invocation.
- A rejected experiment is visible with `--include-rejected` and is excluded
  from future candidates.
- No step claims that submission is approval or that temporary activation is
  confirmation.

## Pitfalls

- Do not optimize against stale facts; reread facts and configuration for each
  new experiment.
- Do not optimize several fields at once merely to save time. Attribution is
  part of the evidence.
- Do not reuse a lesson across a different model, vLLM version, GPU topology,
  or cluster without revalidation.
- Do not report fixture measurements as physical-host results.
