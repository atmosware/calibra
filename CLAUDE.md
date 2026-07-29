# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Calibra is a Claude Code prompt-routing plugin. It intercepts every prompt via a local HTTP proxy, classifies its complexity, and rewrites the `model` field in-flight so cheap prompts use lighter models automatically.

## Key Commands

```bash
# Install / upgrade (also runs automatically on npm install)
node scripts/install.js

# Uninstall
node scripts/uninstall.js

# CLI shorthand
node src/cli.js [install|upgrade|uninstall]

# Recompute ML tier centroids (after editing tools/eval_prompts.jsonl)
node tools/compute_centroids.js

# Golden regression test (heuristic classifier)
node src/ml/classify-core.test.js

# Syntax check all JS
node --check src/saka-proxy.js && node --check src/ml/calibra-ml.js

# Publish
npm version patch && npm publish && git push --follow-tags
```

No test framework, no lint scripts — Node.js built-in `assert` only.

## Architecture

```
Claude Code
    │
    ▼ ANTHROPIC_BASE_URL → http://127.0.0.1:{port}
~/.claude-corp/saka-proxy.js            ← local HTTP proxy (Anthropic Messages API)
    │  1. reads engine flag (heuristic | ml)
    │  2. correctTypos() — hunspell EN/TR typo correction (fail-soft)
    │  3. classifies prompt → tier (light/mid/deep/ultra)
    │  4. downgrade damping — clamps abrupt tier drops (see below)
    │  5. rewrites body.model in-flight
    ▼
Upstream AI server  (CALIBRA_REMOTE_HOST)
```

```
Codex CLI (personal ~/.codex or corp ~/.codex-corp/codex-config)
    │
    ▼ config.toml: model_providers.calibra.base_url → http://127.0.0.1:{fixed-port}/v1
~/.claude-corp/calibra/codex-proxy.js   ← local HTTP proxy (OpenAI Responses API)
    │  always-on background service, autostarted at login (LaunchAgent / Registry Run key) —
    │  config.toml's base_url is static, unlike Claude's per-session env var
    │  1. reads calibra-disabled flag (shared with the Claude side)
    │  2. classifies prompt → tier (light/mid/deep/ultra)
    │  3. rewrites body.model in-flight
    ▼
Real upstream (LiteLLM gateway or api.openai.com), auth forwarded unchanged
```

**Two classification engines:**
- **Heuristic (default):** `calibraClassify()` in `src/ml/classify-core.js` — 5-axis regex scoring, shared by both `saka-proxy.js` and `codex-proxy.js`
- **ML (opt-in, both Claude and Codex):** `classifyML()` in `src/ml/calibra-ml.js` — MiniLM-L6-v2 ONNX + cosine similarity to tier centroids. Each side reads its own engine flag (`calibra-engine` for Claude via `saka-proxy.js`, `calibra-engine-codex` for Codex via `codex-proxy.js`); both fail soft to the heuristic.

**ML long-prompt handling.** MiniLM's trained max sequence length is 256 tokens. `tokenizeChunks()` in `src/ml/tokenizer.js` covers prompts beyond that by pooling instead of tail-truncating: it splits the full token stream into overlapping 256-token windows (56-token stride) and caps at 4 windows via head+tail sampling (keeps first ⌈N/2⌉ + last ⌊N/2⌋ windows, drops the middle — the task statement is usually up front and constraints/edge-cases at the end). `runInference()` in `calibra-ml.js` embeds each window independently (ONNX + mean-pool + L2-normalize), classifies each (linear classifier or centroid path), and returns the **most severe tier** across windows, tie-broken by score — tiering asks whether any part of the prompt looks deep/ultra, not the average. Multi-window runs are tagged with a `+chunked` suffix on `reason` (e.g. `ml-classifier+chunked`) so this is visible in `/calibra status`/logs. Prompts that already fit in 256 tokens produce exactly one window and are unaffected — no behavior or cost change for the common case. `tokenize()` (single-window, tail-truncating) is kept for the training tools (`tools/compute_centroids.js`, `tools/train_tier_classifier.js`), which run over short labeled prompts.

**Downgrade damping (tier hysteresis).** After classification, both proxies clamp
*downgrades* via `applyDamp()` in `src/ml/downgrade-damper.js`, between the tier
decision and the tier→model map. Because each prompt is classified independently, a
sequence like `ultra` then `light` would otherwise snap the model from the heaviest
tier to the cheapest in one step. `applyDamp` limits a new prompt to at most **one
tier below the conversation's previous effective tier**, floored at `mid` (a damped
drop never reaches `light` — `light` is only routed when the previous tier was
already `light`). **Upgrades are never touched** — a prompt classified higher routes
higher immediately. State is a per-conversation store (`calibra-last-tier.json`, keyed
by a hash of the first user message via `keyFrom()`) with a TTL (`CALIBRA_DAMP_TTL_MS`,
default 30 min) so parallel sessions don't clobber and stale threads don't damp. The
**effective (damped)** tier is persisted as the new prev, so a run of cheap prompts
walks down one tier per turn (`ultra→deep→mid→mid`) rather than snapping back. When a
drop is damped, `+damped(prev->eff)` is appended to `reason` for `/calibra status`,
the notify hook, and stderr. Fully fail-soft (missing module / any fs error → raw
tier) and disableable via `CALIBRA_DAMP_DISABLED=1`.

Both engines receive typo-corrected text via `correctTypos()` (from `spellcorrect.js`) before scoring. This is fail-soft: missing dictionaries → correction silently no-ops; missing ONNX model / timeout → ML falls back to heuristic.

**Codex/OpenAI support:** `install.js` detects `~/.codex/config.toml` (personal) and `~/.codex-corp/codex-config/config.toml` (corp) and, for each present, backs up the original file (`config.toml.calibra-backup`), sets `model_provider = "calibra"`, and adds a `[model_providers.calibra]` block pointing `base_url` at a fixed local port — preserving the original provider's `base_url`/`wire_api` as the proxy's real upstream (`~/.claude-corp/calibra/codex-proxy-<personal|corp>.json`, containing `port` + `upstreamHost` + `upstreamPort`). Since `codex-proxy.js` must always be running (static `base_url`, no per-session fallback like the Claude wrapper has), install.js registers a platform-native autostart entry: macOS LaunchAgent (`RunAtLoad` + `KeepAlive`), Windows Registry `Run` key via a hidden `.vbs` launcher. `uninstall.js` reverses all of this — restores `config.toml` from backup, unloads/removes the autostart entry, deletes the proxy config/origin/port files.

**Upstream resolution is persisted (`codex-origin-<personal|corp>.json`).** Once `config.toml` is patched to `model_provider = "calibra"`, the original provider's `base_url`/`wire_api` are no longer discoverable, so install.js records them in an origin file on first patch and recovers them on every re-install (falling back to scanning the leftover `[model_providers.X]` sections / `.calibra-backup` if the origin file is missing). **Without this, a reinstall/upgrade silently resets the upstream to the `api.openai.com` default** — which on a corp network is firewalled/TLS-intercepted and surfaces as a `502 Bad Gateway` from the proxy. The local `base_url` mirrors the real upstream's path prefix (e.g. `/v1`) and `upstreamPort` carries non-443 gateways, so the proxy forwards `req.url` unchanged to `upstreamHost:upstreamPort`.

**Corp CA is baked into the autostart entry.** Corp networks TLS-intercept upstream traffic with a private root CA that Node's bundled store doesn't trust. The Claude side inherits `NODE_EXTRA_CA_CERTS` from `wrapper.sh`, but the Codex proxy is login-launched and inherits no interactive shell env — so if `~/.claude-corp/corp-ca.pem` exists, install.js writes `NODE_EXTRA_CA_CERTS`/`SSL_CERT_FILE`/`CURL_CA_BUNDLE`/`REQUESTS_CA_BUNDLE` into the LaunchAgent's `EnvironmentVariables` (macOS) or the `.vbs` process env (Windows). Missing this env is the other cause of `SELF_SIGNED_CERT_IN_CHAIN` → `502`.

**Toggling Codex routing:** `/calibra on|off|status|toggle` (and the chat forms `enable/disable calibra`) DO work from a `codex-saka` session — but not via a hook, since Codex has none. `codex-proxy.js` intercepts the command itself: `calibraHandleCommand()` matches the prompt in the request body, flips `calibra-disabled-codex` synchronously, then rewrites the request so the real upstream model is instructed (via `injectCommandReply()`, same `instructions`-append trick as the routing note) to relay the confirmation text verbatim on the cheapest tier — the reply still rides through Codex's real Responses-API SSE stream rather than a hand-fabricated one, since Codex's Rust client parses each SSE event into an internal `ResponseItem` via an undocumented schema and a mismatched hand-built stream risks hanging the turn. Gated to `isFreshUserTurn()` so it never fires on tool-call continuations mid-turn. Each environment owns its own flag: `calibra-disabled-claude` (Claude Code, toggled via the `calibra-toggle.js` hook) and `calibra-disabled-codex` (Codex, toggled by `codex-proxy.js`). `/calibra status` in Claude Code shows the state of both.

**Codex ML engine.** The ML engine is available on the Codex side too, switched independently of Claude via its own `calibra-engine-codex` flag. `codex-proxy.js` recognizes `calibra ml on|off`, `calibra rules`, and `enable|disable calibra ml` (slash or plain-text) through `CALIBRA_ML_CMD`, writing the flag with `writeEngine(value, ENGINE_FLAG_PATH_CODEX)`. `calibraRouteOpenAI` is **async**: when the flag is `ml` it `await`s `classifyML()` (same MiniLM path as Claude), else runs the synchronous heuristic — the request handler awaits the decision before forwarding. Fully fail-soft: a failed `require` of `ml/` or any `classifyML` error/timeout falls back to the heuristic. The proxy warms up the ONNX session at startup when the flag is `ml` (avoids first-request cold-start exceeding `CALIBRA_ML_TIMEOUT_MS`).

**Expected noise:** in a corp session (`codex-saka`, which sets `CODEX_HOME` to `~/.codex-corp/codex-config`), Codex also reads `~/.codex/config.toml` as a secondary "project-local" layer (since `$HOME` is an ancestor of any project dir) and logs a warning that `model_provider`/`model_providers`/`notify` are ignored there. Harmless — the corp session's real config is the one `CODEX_HOME` points at, which isn't affected.

## Runtime File Layout

The enterprise wrapper expects the proxy at `~/.claude-corp/saka-proxy.js`. Calibra config, flags, ML assets, and local dependencies live under `~/.claude-corp/calibra/`:

```
~/.claude-corp/
  saka-proxy.js             ← copied from src/ on install
  calibra/
    calibra-models.json     ← tier→model map (never overwritten on upgrade)
    calibra-ml.json         ← ML config (never overwritten on upgrade)
    calibra-disabled-claude ← flag file: routing off for Claude Code when present
    calibra-disabled-codex  ← flag file: routing off for Codex when present
    calibra-engine-codex    ← flag file: 'ml' or 'heuristic' for Codex (absent=heuristic)
    calibra-engine          ← flag file: 'ml' or 'heuristic' (absent=heuristic)
    calibra-proxy-host      ← upstream hostname
    calibra-last-tier.json  ← downgrade-damping store: {convKey→{tier,ts}} (TTL-pruned)
    ml/
      calibra-ml.js         ← ML classifier
      downgrade-damper.js   ← tier hysteresis (clamp downgrades, floor mid)
      classify-core.js      ← shared fast-exits + re-exports correctTypos
      engine-flag.js        ← readEngine/writeEngine
      spellcorrect.js       ← EN/TR hunspell typo correction (fail-soft)
      tokenizer.js          ← BERT WordPiece tokenizer
      vocab.txt             ← bert-base-uncased vocab (30,522 tokens)
      tier-centroids.json   ← 4×384 centroid vectors
    models/
      router.onnx           ← Xenova/all-MiniLM-L6-v2 quantized (~22 MB)
    node_modules/           ← onnxruntime-node + nspell + dictionary-en/en-gb/tr
    package.json
  claude-config/            ← enterprise wrapper (not Calibra's)
```

## Source Layout

| File | Role |
|------|------|
| `src/saka-proxy.js` | Claude-side proxy server (Anthropic Messages API) |
| `src/codex-proxy.js` | Codex-side proxy server (OpenAI Responses API), always-on fixed-port service |
| `src/ml/classify-core.js` | Shared fast-exits, regex constants, `calibraClassify()`, `extractPrompt()`/`extractPromptOpenAI()`, re-exports `correctTypos` |
| `src/ml/calibra-ml.js` | ML engine: ONNX session, chunked-window embedding + pooling, cosine similarity, LRU cache, warmup |
| `src/ml/downgrade-damper.js` | Downgrade damping: pure `damp()` clamp + per-conversation `applyDamp()` store (TTL, atomic writes), `keyFrom()` |
| `src/ml/spellcorrect.js` | Typo correction: nspell + dictionary-en/en-gb/tr, Damerau-Levenshtein, fail-soft |
| `src/ml/tokenizer.js` | BERT WordPiece tokenizer (reads vocab.txt); `tokenize()` single-window truncating (training tools), `tokenizeChunks()` overlapping-window split for long-prompt inference |
| `src/ml/tier-centroids.json` | Baked-in tier centroids (~10 KB) |
| `src/ml/vocab.txt` | bert-base-uncased vocabulary (bundled) |
| `src/calibra-ml.json` | ML metadata: model URL, SHA-256, hiddenSize, maxLength |
| `src/calibra-models.json` | Default tier→model map (Claude models + `openai` block for Codex) |
| `src/hooks/calibra-toggle.js` | Handles `/calibra on\|off\|ml` commands |
| `src/hooks/calibra-notify.js` | Shows routing decision in context bar |
| `src/hooks/calibra-debug.js` | Logs raw hook input to tmpdir |
| `src/commands/calibra.md` | `/calibra` slash command definition |
| `tools/eval_prompts.jsonl` | 759 hand-labeled prompts for centroid computation |
| `tools/compute_centroids.js` | Recomputes tier-centroids.json from eval_prompts.jsonl |
| `scripts/install.js` | Copies files, installs onnxruntime-node, patches settings.json |
| `scripts/uninstall.js` | Removes all installed files and hook entries |

## Configuration

- `~/.claude-corp/calibra/calibra-models.json` — tier→model map (edit to change models)
- `~/.claude-corp/calibra/calibra-ml.json` — ML metadata (rarely edited)
- `CALIBRA_REMOTE_HOST` — upstream AI server hostname
- `CALIBRA_ML_MODEL_PATH` — override ONNX model path (air-gapped installs)
- `CALIBRA_ML_TIMEOUT_MS` — ML inference timeout in ms (default 250)

## Important Invariants

- `calibraClassify()` returns `{tier, score, reason, engine}` — consumed by both proxies AND the notify hook
- `engine-flag.js` uses atomic tmp+rename writes — never write directly to flag files
- `saka-proxy.js`/`codex-proxy.js` must never hard-fail if ML deps are missing — heuristic always works
- Fast-exits (slash command, greeting, trivial, short-conv) run before any ML inference
- `correctTypos()` runs before fast-exits in both engines — corrects EN/TR typos using hunspell, never touches code fences or identifiers
- `calibra-models.json` and `calibra-ml.json` are **never overwritten** on upgrade
- `calibra-disabled-claude`/`calibra-disabled-codex` (routing on/off) and `calibra-engine`/`calibra-engine-codex` (heuristic vs ML) are **separate** per-env flags — toggling one env never affects the other; `/calibra status` shows both
- `engine-flag.js` `readEngine(flagPath?)`/`writeEngine(value, flagPath?)` default to the Claude flag (`ENGINE_FLAG_PATH`); Codex passes `ENGINE_FLAG_PATH_CODEX` explicitly — never hardcode a second path
- `calibraRouteOpenAI()` in `codex-proxy.js` is **async** (ML is async); the request handler must `await` it before forwarding
- `codex-proxy.js` runs on a **fixed** port, unlike `saka-proxy.js` (fresh port per session) — Codex's `config.toml` has a static `base_url`, so the proxy must always be running (autostart, not per-session spawn)
- `runInference()` never raises `maxLength` past 256 — that's MiniLM's own trained max sequence length, not a config knob. Long prompts are handled by chunking into multiple ≤256-token windows (`tokenizeChunks()`), never by a single oversized window
- Chunked inference picks the **most severe tier** across windows (tie-break: higher score) — never an average/mean-pooled decision across windows — since tiering asks whether any part of the prompt is deep/ultra
- Training tools (`compute_centroids.js`, `train_tier_classifier.js`) use `tokenize()` (single-window), not `tokenizeChunks()` — labeled eval prompts are short and centroids/classifiers are fit on single embeddings, so chunking there would change the training distribution
- Downgrade damping (`downgrade-damper.js`) only ever **raises** a downgraded tier back up — it never lowers a tier and never blocks or delays an **upgrade** (`damp()` returns the raw tier unchanged when `new ≥ prev`)
- The damping floor is **mid** — a damped downgrade never routes `light`; `light` is only reached when the previous effective tier was already `light`
- The damping store persists the **effective (damped)** tier as the new prev, never the raw classified tier — else the one-tier-per-turn descent collapses back to a single big drop
- `downgrade-damper.js` is **fail-soft** like the ML path: a missing module or any fs/JSON error routes the raw undamped tier — routing must never hard-fail on damping. `CALIBRA_DAMP_DISABLED=1` bypasses it entirely
- The damping store uses atomic tmp+rename writes (same invariant as `engine-flag.js`) and is keyed per-conversation with a TTL — never a single global last-tier value (parallel Claude/Codex sessions would clobber)

## Improving ML Accuracy

Add labeled prompts to `tools/eval_prompts.jsonl` then:
```bash
node tools/compute_centroids.js    # recomputes src/ml/tier-centroids.json
npm version patch && npm publish   # ships new centroids to users
```
Labels must be human-assigned — do not use `calibraClassify()` to auto-label (circular dependency).
