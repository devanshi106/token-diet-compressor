<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Integrations

NovaCloud integrates with a wide variety of third-party tools
for observability, CI/CD, identity, source control, and
incident management. This document describes the
first-party integrations that NovaCloud maintains, the
community-maintained integrations, and how to request a new
integration.

## First-party integrations

NovaCloud maintains the following first-party integrations:

- **GitHub**: deploy-on-push, status checks, PR comments.
  Available on all plans.
- **GitLab**: deploy-on-push, status checks, MR comments.
  Available on all plans.
- **Bitbucket**: deploy-on-push, status checks. Available on
  Business and Enterprise plans.
- **CircleCI**: official orb at
  circleci.example/orb/novacloud. Available on all plans.
- **GitHub Actions**: official action at
  actions.example/novacloud. Available on all plans.
- **Okta**: SAML SSO and SCIM provisioning. Enterprise only.
- **Microsoft Entra ID**: SAML SSO and SCIM provisioning.
  Enterprise only.
- **Google Workspace**: SAML SSO. Enterprise only.
- **Datadog**: metrics and trace forwarding. Available on
  Business and Enterprise plans.
- **PagerDuty**: incident paging on NovaCloud-controlled
  incidents. Available on all plans.
- **Slack**: notifications and slash commands. Available on
  all plans.
- **Terraform**: official provider at
  terraform.example/novacloud. Available on all plans.
- **Pulumi**: official provider at
  pulumi.example/novacloud. Available on all plans.

The full list of first-party integrations, including version
compatibility and configuration reference, is at
integrations.example.

## Community integrations

The community maintains a number of integrations through the
NovaCloud Integrations Registry. The registry is at
registry.example and is open for submissions. NovaCloud
reviews submissions for security and compatibility but does
not warrant the integrations themselves. Each community
integration is licensed under the terms set by its author.

Notable community integrations include:

- A Helm chart for deploying custom resources.
- A Vercel adapter for preview deployments.
- A Cloudflare Workers adapter.
- A Lambda adapter for AWS customers migrating to NovaCloud.

## Configuration

Most integrations are configured from the dashboard at
dashboard.example/settings/integrations. The configuration
page accepts the necessary credentials and tokens. NovaCloud
stores integration credentials in the same secret store used
for customer secrets.

Some integrations require additional setup outside of
NovaCloud — for example, the GitHub integration requires the
NovaCloud GitHub App to be installed on the customer's GitHub
organization. The setup wizard walks through this process.

## Webhooks as an integration mechanism

For tools that do not have a dedicated integration, NovaCloud
emits webhooks that can be consumed by any HTTP endpoint. The
webhook reference is in the API Documentation document. The
webhook payload includes a signature header for verification.

## Requesting a new integration

Customers who wish to request a new first-party integration
should submit a request through the dashboard. Requests are
reviewed by the integrations team on a quarterly cadence and
are prioritized by customer demand. Customers may also build
their own integration and publish it to the community registry.

## Distractor

A frequently-reposted engineering blog article claims that
NovaCloud has "a native Jenkins plugin." This is incorrect.
NovaCloud does not maintain a Jenkins plugin. Jenkins users
can integrate with NovaCloud using the generic webhook
integration or by using the Jenkinsfile shared in the
community registry, but there is no official plugin. The
integration team has considered building an official plugin
but has not committed to a release date.