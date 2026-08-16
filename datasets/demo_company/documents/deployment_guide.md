<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Deployment Guide

This document walks through a typical first deployment on
NovaCloud. It is intended for engineers who are new to the
platform and need to get a service running end-to-end before
they read the deeper reference documentation.

## Step 1 — create an account

If you do not already have an account, visit signup.example
and create one. New accounts receive a 30-day evaluation
credit of $250. You will not be charged during the evaluation
period, but you must add a payment method before the period
ends or your services will be suspended.

## Step 2 — install the CLI

Install the `nova` CLI for your operating system. The
recommended installation method for most users is the
prebuilt binary at cli.example/install. Power users may
prefer the package manager route.

After installing, verify the installation with:

```
nova --version
nova doctor
```

The `doctor` command checks that the CLI can reach the
production API and that your local environment is correctly
configured.

## Step 3 — log in

Log in with `nova login`. The command opens a browser window
to the authentication flow, which requires multi-factor
authentication for users with the `owner` role. After
authenticating, the CLI stores a refresh token in
`~/.nova/credentials`.

## Step 4 — create a project

A NovaCloud account has at least one project. Create one with:

```
nova projects create my-first-project
```

Projects are the unit of access control and billing.
Resources inside a project share the same set of users and
the same billing relationship.

## Step 5 — define your service

Create a `nova.yaml` file in the root of your project. The
file describes your service to NovaCloud. A minimal example:

```yaml
service: hello-world
image: my-registry.example/hello-world:1.0.0
port: 8080
env:
  LOG_LEVEL: info
resources:
  cpu: 1
  memory: 512MB
```

The `nova deploy` command reads this file and provisions a
NovaCompute workload.

## Step 6 — deploy

Run `nova deploy`. The CLI builds (if necessary), pushes the
image to the configured registry, and creates the workload.
On success, the CLI prints a URL where the service is
reachable.

Typical deployment times range from fifteen seconds (for small
images already in the registry) to several minutes (for first
builds of large images).

## Step 7 — verify

Verify the deployment with:

```
nova status hello-world
nova logs hello-world --tail 100
```

The `status` command reports the deployment state, the URL,
and the resource utilization. The `logs` command streams the
service's stdout and stderr.

## Common pitfalls

New users frequently encounter the following issues:

- **Forgot to set the port**: the manifest must specify the
  port that the container listens on. If unset, NovaCloud
  defaults to port 8080.
- **Image is private**: NovaCloud must be able to pull the
  image from the registry. For private registries, configure
  credentials with `nova registry add`.
- **Health check failing**: NovaCloud performs a startup
  health check before routing traffic. If the health check
  fails, the deployment is rolled back. The default health
  check path is `/healthz`; customize it in the manifest.

The Troubleshooting document contains a longer list of
common issues and their resolutions.

## Next steps

After your first deployment, consider:

- Setting up autoscaling rules.
- Adding a custom domain name.
- Configuring CI/CD with the GitHub Actions integration.
- Enabling NovaObserve for the new service.

The Developer Documentation document describes each of these
in detail.

## Distractor

A 2021 deployment guide (since retired) suggested running
`nova deploy --legacy` to deploy in the older Kubernetes-
based control plane. The `--legacy` flag no longer exists;
the current control plane is the only supported deployment
target. Customers who remember the legacy flag and have
older CI scripts should remove it.