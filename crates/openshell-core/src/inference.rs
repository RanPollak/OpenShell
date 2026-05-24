// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Auth header abstraction
// ---------------------------------------------------------------------------

/// How to inject an API key on outgoing inference requests.
///
/// Defined in `openshell-core` so both `openshell-router` (which applies it)
/// and `openshell-server` / `openshell-sandbox` (which resolve it from
/// provider metadata) can share the same type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthHeader {
    /// `Authorization: Bearer <key>`
    Bearer,
    /// Custom header name (e.g. `x-api-key` for Anthropic).
    Custom(&'static str),
}

// ---------------------------------------------------------------------------
// Model-source policy
// ---------------------------------------------------------------------------

/// Per-route policy that decides how the inference router treats the client's
/// `model` field when forwarding a request to the upstream provider.
///
/// Silently overwriting the client-supplied model masks typos and stale model
/// references — the upstream provider's "unknown model" error never reaches
/// the caller. Making the policy explicit lets operators pick between
/// provider-swap convenience and request-integrity guarantees without
/// changing the default.
///
/// In all three modes, a missing or empty client model falls back to
/// `route.model`, because every supported upstream provider requires a
/// non-empty `model` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSource {
    /// Existing behaviour: the route's configured model wins, even when the
    /// client sent a different value. A mismatch is logged at `warn` level so
    /// operators can spot accidental typos without rejecting the request.
    /// This is the default to keep the historical provider-swap workflow
    /// working without explicit opt-in.
    #[default]
    Router,
    /// The client-supplied model is forwarded to the upstream provider
    /// unchanged. Restores the standard "unknown model" error surface so
    /// caller-side typos and stale references fail loudly.
    Caller,
    /// Like `Caller`, but the router rejects the request with an error when
    /// the client-supplied model does not match `route.model`. Use this when
    /// the operator wants to enforce a single configured model.
    Matching,
}

impl ModelSource {
    /// Parse a case-insensitive token (`router` / `caller` / `matching`) into
    /// a [`ModelSource`]. Returns `None` for any other value so callers can
    /// surface a per-flag validation error.
    pub fn parse(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "router" => Some(Self::Router),
            "caller" => Some(Self::Caller),
            "matching" => Some(Self::Matching),
            _ => None,
        }
    }

    /// Canonical token used by the CLI, gRPC string fields, and the YAML
    /// router config.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::Caller => "caller",
            Self::Matching => "matching",
        }
    }
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Inference provider profiles
// ---------------------------------------------------------------------------

/// Static metadata describing how to talk to a specific inference provider's API.
///
/// This is the single source of truth for provider-specific inference knowledge:
/// default endpoint, supported protocols, credential key lookup order, auth
/// header style, default headers, and allowed client-supplied passthrough
/// headers.
///
/// This is separate from [`openshell_providers::ProviderPlugin`] which handles
/// credential *discovery* (scanning env vars). `InferenceProviderProfile` handles
/// how to *use* discovered credentials to make inference API calls.
pub struct InferenceProviderProfile {
    pub provider_type: &'static str,
    pub default_base_url: &'static str,
    pub protocols: &'static [&'static str],
    /// Credential map key names to search for the API key, in priority order.
    pub credential_key_names: &'static [&'static str],
    /// Config map key names to search for a base URL override, in priority order.
    pub base_url_config_keys: &'static [&'static str],
    /// Auth header style for outgoing requests.
    pub auth: AuthHeader,
    /// Default headers injected on every outgoing request.
    pub default_headers: &'static [(&'static str, &'static str)],
    /// Client-supplied headers that may be forwarded to the upstream backend.
    ///
    /// Header names must be lowercase and must not include auth headers.
    pub passthrough_headers: &'static [&'static str],
}

const OPENAI_PROTOCOLS: &[&str] = &[
    "openai_chat_completions",
    "openai_completions",
    "openai_responses",
    "model_discovery",
];

const ANTHROPIC_PROTOCOLS: &[&str] = &["anthropic_messages", "model_discovery"];

static OPENAI_PROFILE: InferenceProviderProfile = InferenceProviderProfile {
    provider_type: "openai",
    default_base_url: "https://api.openai.com/v1",
    protocols: OPENAI_PROTOCOLS,
    credential_key_names: &["OPENAI_API_KEY"],
    base_url_config_keys: &["OPENAI_BASE_URL"],
    auth: AuthHeader::Bearer,
    default_headers: &[],
    passthrough_headers: &["openai-organization", "x-model-id"],
};

static ANTHROPIC_PROFILE: InferenceProviderProfile = InferenceProviderProfile {
    provider_type: "anthropic",
    default_base_url: "https://api.anthropic.com/v1",
    protocols: ANTHROPIC_PROTOCOLS,
    credential_key_names: &["ANTHROPIC_API_KEY"],
    base_url_config_keys: &["ANTHROPIC_BASE_URL"],
    auth: AuthHeader::Custom("x-api-key"),
    default_headers: &[("anthropic-version", "2023-06-01")],
    passthrough_headers: &["anthropic-version", "anthropic-beta"],
};

static NVIDIA_PROFILE: InferenceProviderProfile = InferenceProviderProfile {
    provider_type: "nvidia",
    default_base_url: "https://integrate.api.nvidia.com/v1",
    protocols: OPENAI_PROTOCOLS,
    credential_key_names: &["NVIDIA_API_KEY"],
    base_url_config_keys: &["NVIDIA_BASE_URL"],
    auth: AuthHeader::Bearer,
    default_headers: &[],
    passthrough_headers: &["x-model-id"],
};

/// Look up the inference provider profile for a given provider type.
///
/// Returns `None` for provider types that don't support inference routing
/// (e.g. `github`, `gitlab`, `outlook`).
pub fn profile_for(provider_type: &str) -> Option<&'static InferenceProviderProfile> {
    match provider_type.trim().to_ascii_lowercase().as_str() {
        "openai" => Some(&OPENAI_PROFILE),
        "anthropic" => Some(&ANTHROPIC_PROFILE),
        "nvidia" => Some(&NVIDIA_PROFILE),
        _ => None,
    }
}

/// Derive the [`AuthHeader`] and default headers for a provider type string.
///
/// This is a convenience wrapper around [`profile_for`] for callers that only
/// need the auth/header information (e.g. the sandbox bundle-to-route
/// conversion).
pub fn auth_for_provider_type(provider_type: &str) -> (AuthHeader, Vec<(String, String)>) {
    let (auth, headers, _) = route_headers_for_provider_type(provider_type);
    (auth, headers)
}

/// Derive routing header policy for a provider type string.
///
/// Returns the auth injection mode, route-level default headers, and the
/// allowed client-supplied passthrough headers for `inference.local`.
pub fn route_headers_for_provider_type(
    provider_type: &str,
) -> (AuthHeader, Vec<(String, String)>, Vec<String>) {
    profile_for(provider_type).map_or_else(
        || (AuthHeader::Bearer, Vec::new(), Vec::new()),
        |profile| {
            let headers = profile
                .default_headers
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            let passthrough_headers = profile
                .passthrough_headers
                .iter()
                .map(|name| (*name).to_string())
                .collect();
            (profile.auth.clone(), headers, passthrough_headers)
        },
    )
}

// ---------------------------------------------------------------------------
// Protocol normalization
// ---------------------------------------------------------------------------

/// Normalize a list of protocol strings: trim, lowercase, deduplicate, skip empty.
pub fn normalize_protocols(protocols: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();

    for protocol in protocols {
        let candidate = protocol.trim().to_ascii_lowercase();
        if candidate.is_empty() {
            continue;
        }
        if seen.insert(candidate.clone()) {
            normalized.push(candidate);
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_deduplicates() {
        let input = vec![
            "OpenAI_Chat_Completions".to_string(),
            " openai_chat_completions ".to_string(),
            "anthropic_messages".to_string(),
        ];
        let result = normalize_protocols(&input);
        assert_eq!(
            result,
            vec!["openai_chat_completions", "anthropic_messages"]
        );
    }

    #[test]
    fn skips_empty_and_whitespace() {
        let input = vec![String::new(), "  ".to_string(), "valid".to_string()];
        let result = normalize_protocols(&input);
        assert_eq!(result, vec!["valid"]);
    }

    #[test]
    fn empty_input() {
        let result = normalize_protocols(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn profile_for_known_types() {
        assert!(profile_for("openai").is_some());
        assert!(profile_for("anthropic").is_some());
        assert!(profile_for("nvidia").is_some());
        assert!(profile_for("OpenAI").is_some()); // case insensitive
    }

    #[test]
    fn profile_for_unknown_types() {
        assert!(profile_for("github").is_none());
        assert!(profile_for("gitlab").is_none());
        assert!(profile_for("unknown").is_none());
    }

    #[test]
    fn auth_for_anthropic_uses_custom_header() {
        let (auth, headers) = auth_for_provider_type("anthropic");
        assert_eq!(auth, AuthHeader::Custom("x-api-key"));
        assert!(headers.iter().any(|(k, _)| k == "anthropic-version"));
    }

    #[test]
    fn route_headers_for_openai_include_passthrough_headers() {
        let (_, _, passthrough_headers) = route_headers_for_provider_type("openai");
        assert!(
            passthrough_headers
                .iter()
                .any(|name| name == "openai-organization")
        );
        assert!(passthrough_headers.iter().any(|name| name == "x-model-id"));
    }

    #[test]
    fn route_headers_for_anthropic_include_passthrough_headers() {
        let (_, _, passthrough_headers) = route_headers_for_provider_type("anthropic");
        assert!(
            passthrough_headers
                .iter()
                .any(|name| name == "anthropic-version")
        );
        assert!(
            passthrough_headers
                .iter()
                .any(|name| name == "anthropic-beta")
        );
    }

    #[test]
    fn auth_for_openai_uses_bearer() {
        let (auth, headers) = auth_for_provider_type("openai");
        assert_eq!(auth, AuthHeader::Bearer);
        assert!(headers.is_empty());
    }

    #[test]
    fn auth_for_unknown_defaults_to_bearer() {
        let (auth, headers) = auth_for_provider_type("unknown");
        assert_eq!(auth, AuthHeader::Bearer);
        assert!(headers.is_empty());
    }

    #[test]
    fn route_headers_for_unknown_are_empty() {
        let (auth, headers, passthrough_headers) = route_headers_for_provider_type("unknown");
        assert_eq!(auth, AuthHeader::Bearer);
        assert!(headers.is_empty());
        assert!(passthrough_headers.is_empty());
    }
}
