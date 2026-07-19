# NixClaw integration contract

This repository supplies the untrusted Hermes-side client and the host-side
benchmark executable. The privileged broker and activator are implemented in
`benthecarman/nemoclaw-nix`.

## Package handoff

Consume this flake from the privileged NixOS configuration, follow its
`nixpkgs` input, and add `inputs.nixclaw.overlays.default`. The resulting
`pkgs.nixclaw` derivation supplies the Hermes and host commands without mutable
Python installation:

```console
nix build .#nixclaw
```

The package exposes `nixclaw-agent` inside the Hermes sandbox and
`nixclaw-bench`, `nixclaw-policy`, and `nixclaw-adversarial` to the relevant
host or sandbox services. Schemas, skills, policy templates, and workloads are
available under `${pkgs.nixclaw}/share/nixclaw`. Do not install the project
with mutable `pip` on the Spark.

The Hermes sandbox needs these variables:

```text
NIXCLAW_BROKER_URL=https://broker.internal:8443
NIXCLAW_STATE_DIR=/sandbox/state/nixclaw
NIXCLAW_BROKER_CREDENTIAL=openshell:resolve:env:NIXCLAW_BROKER_TOKEN
```

The real broker credential belongs to the OpenShell provider. The sandbox
sends only the placeholder, which the proxy resolves after its REST policy
admits the request.

## Broker behavior

The source contract is `schemas/nixclaw/v1`. The broker must reject unknown
fields, stale generations, unsupported tunables, redirects, and duplicate
requests with conflicting payloads. A repeated idempotency key with an
identical payload returns the original experiment.

The optimizer stops after submission. It never calls approval or activation;
those operations remain available only through the host activator socket.

## Activator benchmark call

For each temporary candidate, the operator runs the same workload directly
against the stable replica and the drained canary. Bypassing the normal router
keeps measurements attributable to one node. Each result names that node and
supplies generation-specific host signals collected from systemd, the kernel
log, and GPU telemetry.

```console
nixclaw-bench run \
  --endpoint http://127.0.0.1:8000 \
  --model served-model \
  --workload workloads/agent-tools.json \
  --environment-fingerprint sha256:... \
  --node-id nixos-s4 \
  --generation /nix/store/...-nixos-system \
  --profile-hash sha256:... \
  --health-signals host-signals.json \
  --output candidate.json

nixclaw-bench compare \
  --baseline baseline.json \
  --candidate candidate.json \
  --output decision.json

nixclawctl record-results EXPERIMENT_ID \
  --baseline baseline.json \
  --candidate candidate.json \
  --decision decision.json
```

`host-signals.json` follows `workloads/host-signals.example.json`. It records
the health failure, restart, OOM, and NCCL error counts plus whether critical
memory pressure occurred.

An operator attaches both benchmark files and the decision through the
activator's Unix socket before confirmation. The activator verifies them
against `experiment-results.schema.json`, including that the baseline result
came from an advertised stable replica and the candidate result came from the
approved canary. Confirmation promotes an accepted generation to the stable
replicas and restores normal routing. Rejection or lease expiry rolls back only
the canary. Afterward Hermes runs `nixclaw-agent experiments sync ID`; the local
store promotes accepted evidence or records negative evidence.

## OpenShell policy

Render `openshell/policy.yaml.in` with immutable Nix store paths and concrete
broker/inference destinations. Apply it at sandbox creation because filesystem,
Landlock, and process fields are static.

After onboarding, run `nixclaw-adversarial` inside the sandbox. It must permit
the facts route while denying the nonexistent activator route, arbitrary
egress, the activator socket, Nix daemon socket, Docker socket, and host NixOS
configuration.

## DGX Spark acceptance

The current target, `hackathon@nixos-s6`, is ARM64 NixOS with one NVIDIA GB10
SM121 GPU and approximately 121 GiB system memory. Before live acceptance:

1. Person 1 consumes `pkgs.nixclaw` and its agreed JSON Schemas through Nix.
2. Deploy the broker, activator, vLLM 0.25.1 service, Hermes, and OpenShell.
3. Render and apply the concrete OpenShell policy.
4. Run the two workload manifests against the configurable served model.
5. Complete one experiment through approval, temporary activation, comparison,
   confirmation or rollback, and lesson synchronization.
6. Start a fresh Hermes session and verify the validated lesson is retrieved.
7. Run the live adversarial command and inspect OpenShell denial logs.

Fixture results are never substitutes for this physical-host acceptance.
