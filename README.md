# Calibra

Automatic per-prompt model routing for Claude Code. Calibra intercepts every prompt via a local HTTP proxy, classifies its complexity, and rewrites the `model` field in-flight — so cheap prompts use lighter models automatically, without any manual switching.

---

## Tiers

| Tier | Default Model | When |
|------|--------------|------|
| `light` | Haiku | Greetings, recall/lookup, trivial one-liners (add log, rename, fix typo, format) |
| `mid` | Sonnet | Concrete single-component work (fix, build, write, implement, explain) |
| `deep` | Opus | Synthesis and judgment scoped to one system (design, audit, optimize, diagnose) |
| `ultra` | Opus | Multi-system or org-wide programs (≥2 named subsystems, comprehensive, multi-quarter) |

---

## How It Works

### 1. The Proxy

Calibra installs a local HTTP proxy (`saka-proxy.js`) and sets `ANTHROPIC_BASE_URL` to point at it. Every API request from Claude Code passes through before reaching the upstream server.

```
Claude Code
    │
    ▼  ANTHROPIC_BASE_URL → http://127.0.0.1:{port}
saka-proxy.js                   ← reads prompt, classifies, rewrites model
    │
    ▼  CALIBRA_REMOTE_HOST
Upstream AI Server
```

See [`docs/diagrams/proxy-architecture.drawio`](docs/diagrams/proxy-architecture.drawio) for the full intercept flow.

Two classification engines are available. The active engine is controlled by the `calibra-engine` flag file (absent = heuristic).

---

### 2. Heuristic Engine (default)

The heuristic engine scores every prompt across **five independent axes** — no signal appears in more than one axis — then maps the total to a tier.

**Step-by-step:**

0. **Typo correction (`correctTypos`)** — applied once before scoring. Hunspell dictionaries (`dictionary-en`, `dictionary-en-gb`, `dictionary-tr`) plus Damerau-Levenshtein distance correct single-character typos (e.g. `secuity` → `security`, `analz` → `analiz`). Code fences and identifiers are never touched. Fail-soft: if dictionaries are missing, this step is a no-op.

1. **Early exits** — checked in order, short-circuit immediately:
   - Greeting or social acknowledgement → `light`
   - Trivial one-liner (add console.log, rename variable, fix typo, add null check) → `light`
   - Slash command (`/`) → `mid`
   - Short prompt (≤55 chars) with no actionable signal → `light`

2. **5-axis scoring:**

   | Axis | Signal | Points |
   |------|--------|--------|
   | **1 — Length** | > 500 chars | +3 |
   | | > 200 chars | +2 |
   | | ≥ 80 chars | +1 |
   | **2 — Intent** | Deep verbs: `architect`, `design`, `analyse`, `audit`, `investigate`, `diagnose`, `review`, `optimize`, `harden`, `evaluate`, `compare`, `strategy`, `plan`, … | +3 |
   | | Mid verbs: `implement`, `build`, `create`, `write`, `fix`, `debug`, `add`, `update`, `migrate`, `explain`, `configure`, … | +1 |
   | **3 — Scope** | Breadth words: `comprehensive`, `entire`, `full`, `end-to-end`, `exhaustive`, `detailed`, `overall`, `holistic`, `company-wide`, `org-wide`, `genelinde`, … | +2 |
   | **4 — Domain** | Technical vocabulary: `distributed`, `microservices`, `kubernetes`, `graphql`, `grpc`, `authentication`, `event-driven`, `sharding`, `circuit breaker`, `scalability`, … | +2 |
   | **5 — Structure** | Multiple code blocks or block > 52 lines | +2 |
   | | Single code block | +1 |
   | | Step-by-step / multi-part markers | +1 |

3. **Floor rules:**
   - Any deep-intent verb → minimum `deep` regardless of score
   - Any mid-intent verb → minimum `mid`

4. **Threshold mapping** (max realistic score ≈ 13):

   ```
   score 0–2, no intent      → light
   score 0–7, mid intent     → mid
   deep intent present       → deep   (floor)
   score 8+                  → ultra
   ```

5. **Model rewrite** — `model` field in the request body is replaced with the tier's configured model before forwarding.

See [`docs/diagrams/heuristic-engine.drawio`](docs/diagrams/heuristic-engine.drawio) for the full flow.

---

### 3. ML Engine (opt-in — `/calibra ml on`)

The ML engine uses a **rule-first cascade**: deterministic rules handle the clearly-decidable cases; a MiniLM neural model owns the genuinely ambiguous residual under a cost objective.

**Step-by-step:**

1. **Rule layer (`ruleClassify`)** — evaluated in order, first match wins and commits immediately:

   | Rule | Condition | Result |
   |------|-----------|--------|
   | 1 | Empty prompt or slash command | `mid` |
   | 2 | Pure greeting / social | `light` |
   | 3 | Trivial single-edit EN/TR (add null check, rename symbol, fix indentation, add return statement, add null guard, "yeniden isimlendir", "girintiyi duzelt", …) | `light` |
   | 4 | ≥2 breadth words **or** 3+ item enumeration **or** ≥2 distinct named subsystems joined by a coordinator (`auth and payments`, `mobile app and backend`, …) | `ultra` |
   | 5–7 | Intent verb (deep/mid) or short-no-signal | **not confident → defer to ML** |

   Rules 1–4 are 100% precise on every eval set and commit without calling the model. Rules 5–7 defer because an intent verb alone does not pin the tier — that is the irreducible ambiguity the ML must own.

2. **Typo correction (`correctTypos`)** — same EN/TR hunspell correction as the heuristic engine, applied before the rule layer. The ONNX embedding also receives corrected text, so a typo variant and its correct form produce the same (or very close) sentence vector.

3. **MiniLM ONNX pipeline** (rules 5–7 residual only):
   1. Tokenize prompt with BERT WordPiece (`bert-base-uncased`). MiniLM's trained max sequence length is 256 tokens, so a longer prompt is split into overlapping 256-token windows (56-token stride) instead of tail-truncated — up to 4 windows, keeping the head + tail windows and dropping the middle if there are more (task statement is usually up front, constraints/edge-cases at the end).
   2. Run `all-MiniLM-L6-v2` ONNX on each window → `last_hidden_state` [1 × seq × 384]
   3. Mean-pool with attention mask → sentence vector [384] per window; L2-normalize
   4. Append lexical feature axes (`deepC`, `midC`, `scopeC`, `domainC`) and classify each window independently
   5. Ordinal regression head → tier posterior distribution [light, mid, deep, ultra], per window
   6. `expectedCostDecision(policy I)` — choose the tier minimising **expected routing cost** over the posterior, not raw argmax, per window. The cost matrix biases the ambiguous boundary toward the safer tier (never severely under-route). Across windows, the **most severe tier wins** (tie-break: higher score) — a prompt is as complex as its hardest part, not its average. A single window (the common case) skips straight to that window's decision.

4. **Fail-soft** — if the ONNX model is absent, times out (`CALIBRA_ML_TIMEOUT_MS`), or throws, the system silently falls back to the heuristic engine. No error is shown to the user.

See [`docs/diagrams/ml-engine.drawio`](docs/diagrams/ml-engine.drawio) for the full cascade.

**Accuracy:** ~93% on the dev benchmark; **~88% on independent holdouts** labeled by separate models. The ~5% gap is the irreducible label-noise ceiling — terse, typo-heavy, and multi-system prompts sit at genuine tier boundaries where labelers themselves disagree. See [`docs/RESULTS.md`](docs/RESULTS.md) for the full cross-labeler analysis.

---

## Requirements

- Node.js ≥ 18
- Claude Code CLI
- An enterprise wrapper that sets `CALIBRA_REMOTE_HOST` and `ANTHROPIC_BASE_URL` before launching Claude Code

---

## Installation

**Option A — npx (recommended)**

```sh
npx calibra install
```

**Option B — global install**

```sh
npm install -g calibra
```

If you get `EACCES: permission denied`:

```sh
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
npm install -g calibra
```

> If you use **nvm** or **fnm**, global installs already work without this step.

The postinstall script:

1. Copies `saka-proxy.js` to `~/.claude-corp/`
2. Copies hooks to `~/.claude/hooks/`
3. Copies the `/calibra` command to `~/.claude/commands/`
4. Creates `~/.claude-corp/calibra/calibra-models.json` (first install only — never overwritten on upgrade)
5. Copies ML runtime files (including `spellcorrect.js`) to `~/.claude-corp/calibra/ml/`
6. Installs `onnxruntime-node`, `nspell`, `dictionary-en`, `dictionary-en-gb`, `dictionary-tr` into `~/.claude-corp/calibra/node_modules/` (skips packages already present)
7. Registers hooks in `~/.claude/settings.json`

---

## Configuration

### Model tiers — `~/.claude-corp/calibra/calibra-models.json`

Edit to change which model each tier uses:

```json
{
  "light": "claude-haiku-4-5-20251001",
  "mid":   "claude-sonnet-4-6",
  "deep":  "claude-opus-4-7",
  "ultra": "claude-opus-4-7",
  "nonAnthropicModels": []
}
```

Never overwritten on upgrade. `nonAnthropicModels` lists model IDs that need special request sanitisation (strip thinking blocks, set min `max_tokens`).

### Remote host

```sh
export CALIBRA_REMOTE_HOST="your-litellm-server.example.com"
```

### ML environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CALIBRA_ML_MODEL_PATH` | Path to a local `.onnx` file (air-gapped installs) | `~/.claude-corp/calibra/models/router.onnx` |
| `CALIBRA_ML_TIMEOUT_MS` | Max inference time before falling back to heuristic | `250` |

**Spell-check dependencies** (`nspell`, `dictionary-en`, `dictionary-en-gb`, `dictionary-tr`) are installed automatically into `~/.claude-corp/calibra/node_modules/` by the install script. If they are absent, typo correction silently no-ops — routing still works.

---

## Usage

Calibra runs silently. A context note appears on each prompt:

```
calibra: >> claude-sonnet-4-6 . mid
```

### `/calibra` command

```
/calibra status    → show routing state for Claude Code and Codex separately
/calibra on        → enable routing for Claude Code
/calibra off       → disable routing for Claude Code (original model used)
/calibra toggle    → flip Claude Code state
/calibra ml on     → switch to ML engine (downloads model on first use)
/calibra ml off    → switch back to heuristic
/calibra rules     → alias for ml off
```

Natural-language phrases also work: `disable calibra`, `enable calibra`.

Routing state is **per-environment** — Claude Code and Codex each have their own flag. `/calibra on|off` from a Claude Code session only affects Claude Code; the equivalent phrases in Codex only affect Codex.

### Codex CLI users

Codex has no slash command system — `/calibra` will produce `Unrecognized command '/calibra'` in the Codex TUI. Use plain-text phrases instead; `codex-proxy.js` intercepts them at the wire before they reach the model:

| Intent | Type this in Codex |
|--------|-------------------|
| Enable routing | `enable calibra` |
| Disable routing | `disable calibra` |
| Check status | `status calibra` |
| Turn on | `turn on calibra` |
| Turn off | `turn off calibra` |
| Enable ML engine | `calibra ml on` (or `enable calibra ml`) |
| Disable ML engine | `calibra ml off` (or `disable calibra ml`, `calibra rules`) |

Each environment maintains its own enable/disable **and** engine state — toggling in Codex does not affect a running Claude Code session, and vice versa. Codex reads its own `calibra-engine-codex` flag; Claude Code reads `calibra-engine`.

> **Note:** The ML engine is available on **both** Claude Code and Codex, each switched independently. As on the Claude side, Codex's ML path is fully fail-soft — if the ONNX model is absent, times out, or errors, it silently falls back to the heuristic. The routing decision is shown inline at the top of each Codex reply: `» [calibra: <model> · <tier>]`.

---

## ML Engine: First Activation

The first time you run `/calibra ml on`, a ~22 MB quantized ONNX model is downloaded to `~/.claude-corp/calibra/models/router.onnx` and verified against a SHA-256 checksum.

---

## Improving ML Accuracy

Add labeled prompts to `tools/eval_prompts.jsonl` then retrain:

```sh
node tools/train_tier_classifier.js     # re-fits the ordinal head
node tools/tune_thresholds.js tools/final_holdout_opus_500.jsonl --write
node tools/evaluate_classifier.js tools/adversarial_eval.jsonl   # honest number
```

**Labeling rubric (keep human-assigned, never circular):**

- `light` — no judgment: recall/lookup, single-statement mechanical edit, or social
- `mid` — one bounded component with chosen logic: implement/fix/debug/refactor a single function or feature
- `deep` — synthesis/judgment scoped to ONE system: design/architect/analyze/audit/optimize
- `ultra` — multi-system OR org/platform-wide OR long program: ≥2 named subsystems joined, comprehensive/company-wide, multi-quarter

**Evaluation protocol — keep eyes separate:**

| Role | File | Used for |
|------|------|----------|
| fit | `eval_prompts.jsonl` + `targeted_train_opus_800.jsonl` | gradient fit |
| dev | `final_holdout_opus_500.jsonl` | threshold tuning only |
| test | `calibra_eval_set.jsonl`, `adversarial_eval.jsonl` | report only — never trained/tuned on |

---

## Upgrade

```sh
npx calibra upgrade
```

- `saka-proxy.js` and hooks are updated
- `calibra-models.json` is **never overwritten**
- `calibra-ml.json` is **never overwritten**

---

## Uninstall

```sh
npx calibra uninstall
```

Removes all installed files, hooks, and hook entries from `settings.json`.

---

## Files Installed

| File | Location | Purpose |
|------|----------|---------|
| `saka-proxy.js` | `~/.claude-corp/` | Proxy — classifies prompts, rewrites model |
| `calibra-models.json` | `~/.claude-corp/calibra/` | Tier → model mapping (user config) |
| `calibra-ml.json` | `~/.claude-corp/calibra/` | ML metadata and local model settings |
| `calibra-disabled-claude` | `~/.claude-corp/calibra/` | Flag file — routing off for Claude Code when present |
| `calibra-disabled-codex` | `~/.claude-corp/calibra/` | Flag file — routing off for Codex when present |
| `ml/` | `~/.claude-corp/calibra/` | ML classifier, tokenizer, vocab, centroids, spellcorrect |
| `models/router.onnx` | `~/.claude-corp/calibra/` | Downloaded ONNX model (ML mode) |
| `calibra-notify.js` | `~/.claude/hooks/` | Shows routing decision in context bar |
| `calibra-debug.js` | `~/.claude/hooks/` | Logs raw hook input to `<tmpdir>/calibra-debug.log` |
| `calibra-toggle.js` | `~/.claude/hooks/` | Handles Claude Code toggle commands |
| `calibra.md` | `~/.claude/commands/` | `/calibra` slash command definition |

---

## Publishing a New Version

```sh
npm version patch   # or minor / major
npm publish
git push && git push --tags
```

---

## Platforms

| Platform | Status |
|----------|--------|
| macOS | Supported |
| Linux | Supported |
| Windows (native) | Supported |

---

## Quick Reference

| Action | Command |
|--------|---------|
| Install | `npx calibra install` |
| Check status | `/calibra status` in Claude Code |
| Upgrade | `npx calibra upgrade` |
| Uninstall | `npx calibra uninstall` |
| Enable ML engine | `/calibra ml on` |
| Disable routing | `/calibra off` |
