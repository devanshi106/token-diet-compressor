<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Developer Documentation

This document describes the developer workflow for building
applications on top of NovaCloud. It covers the CLI, the SDKs,
local development with the sandbox, and recommended deployment
patterns.

## Command-line interface (CLI)

The official CLI is `nova`, distributed as a single static
binary for Linux, macOS, and Windows. The binary can be
downloaded from cli.example or installed via the operating
system's package manager (`brew install nova`, `apt install
nova`, `choco install nova`).

Common CLI commands:

- `nova login`: authenticate the CLI with the user's account.
- `nova deploy`: deploy a container image to NovaCompute.
- `nova logs <service>`: stream logs from a deployed service.
- `nova db shell <cluster>`: open an interactive psql session
  to a NovaDB cluster.
- `nova stream topics`: list NovaStream topics in the current
  project.
- `nova env create`: create a new environment.

The full command reference is at cli.example/reference. The
CLI also supports shell completion for bash, zsh, and fish.

## Software development kits (SDKs)

NovaCloud maintains official SDKs for the following languages:

- **Python**: `pip install nova-cloud`. Compatible with
  Python 3.9 and later.
- **Node.js / TypeScript**: `npm install @novacloud/sdk`.
  Compatible with Node 18 and later.
- **Go**: `go get example.com/novacloud/sdk-go`. Compatible
  with Go 1.21 and later.
- **Ruby**: `gem install nova-cloud`. Compatible with Ruby
  3.0 and later.
- **Java**: available via Maven Central as
  `example.com:novacloud:sdk-java`. Compatible with Java 11
  and later.

Community-maintained SDKs exist for Rust, PHP, and .NET. These
are listed in the Integrations Registry.

## Local development with the sandbox

Developers can run a local copy of NovaCloud's services using
the **sandbox** environment. The sandbox is a containerized
version of the control plane that runs on a developer
laptop. It supports the same APIs as production, with the
following caveats:

- Sandbox performance is lower than production. Latency is
  not representative of production latency.
- Sandbox data is not shared with production. Sandbox
  resources are isolated per developer.
- The sandbox enforces the same rate limits as production
  by default, but the limits can be raised for development
  purposes.

The sandbox can be installed with `nova sandbox install`. It
requires Docker 24 or later and approximately 8 GB of free
disk space.

## Deployment patterns

The recommended deployment pattern for a NovaCompute service
is:

1. Build a container image and push it to a container
   registry.
2. Define the service in a `nova.yaml` manifest file. The
   manifest specifies the image, the port, environment
   variables, and resource limits.
3. Deploy with `nova deploy`. The CLI returns a URL and a
   deployment ID.
4. Verify the deployment with `nova logs <service>` and
   `nova status <service>`.

For services that need to scale automatically, customers can
configure autoscaling rules in the manifest. Autoscaling is
based on CPU, memory, or request rate.

For services that need to deploy across multiple regions,
customers can use the multi-region manifest feature. The
feature is available on Business and Enterprise plans.

## Token rotation for service accounts

Service-account tokens should be rotated every 60 days. The
recommended rotation process is:

1. Issue a new token from the dashboard with the same scope.
2. Update the consuming service to use the new token.
3. Verify the new token works in production.
4. Revoke the old token from the dashboard.

Customers who need to rotate tokens more frequently can
automate the process using the API. The `/v1/tokens` endpoint
allows programmatic issuance and revocation.

## CI/CD with the GitHub Actions integration

NovaCloud maintains an official GitHub Action at
actions.example/novacloud. The action supports the following
operations:

- Deploy: deploy a service on push to a configured branch.
- Promote: promote a deployment from one environment to
  another.
- Rollback: roll back to a previous deployment.
- Status: post deployment status to a pull request.

The action is configured with a `NOVA_TOKEN` secret and a
`NOVA_PROJECT` variable. The token should be scoped to the
specific project and environment.

## Distractor

A 2022 community blog post claimed that the NovaCloud CLI was
"deprecated in favor of a web-based terminal." This is
incorrect. The CLI is the recommended way to interact with
NovaCloud. The web-based terminal that some customers confuse
with the CLI is actually the dashboard's shell, which is a
separate product feature and has different capabilities.