'use strict';

// downgrade-damper.js — tier hysteresis for Calibra routing.
//
// Calibra classifies each prompt independently, so a sequence like
// ultra → light snaps the model from the heaviest tier to the cheapest in one
// step. That drop is too abrupt mid-conversation. This module damps *downgrades*
// only: a new prompt may fall at most one tier below the previous *effective*
// (already-damped) tier, and never below `mid`. Upgrades are never touched — a
// prompt classified higher routes higher immediately.
//
//   ultra → light  ⇒ deep      deep → light ⇒ mid
//   mid   → light  ⇒ mid        light → light ⇒ light
//   ultra → deep   ⇒ deep       ultra → mid  ⇒ deep (only one step down)
//   mid   → ultra  ⇒ ultra (upgrade, untouched)
//
// Fully fail-soft: any fs/JSON error returns the raw tier undamped. Routing must
// never hard-fail on this (CLAUDE.md invariant).

const fs   = require('fs');
const os   = require('os');
const path = require('path');
const crypto = require('crypto');

const ORDER = { light: 0, mid: 1, deep: 2, ultra: 3 };
const NAMES = ['light', 'mid', 'deep', 'ultra'];
const MID   = 1;

const STORE_PATH = path.join(os.homedir(), '.claude-corp', 'calibra', 'calibra-last-tier.json');
const TTL_MS = Number(process.env.CALIBRA_DAMP_TTL_MS) || 30 * 60 * 1000; // 30 min

// Pure clamp. new >= prev → upgrade/same, untouched. Otherwise floor at
// max(prev-1, mid): one step down, never below mid.
function damp(newTier, prevTier) {
  const n = ORDER[newTier];
  const p = ORDER[prevTier];
  if (n == null || p == null) return newTier; // unknown tier — don't touch
  if (n >= p) return newTier;
  const floor = Math.max(p - 1, MID);
  return NAMES[Math.max(n, floor)];
}

// Stable 16-hex key from the conversation head text the caller supplies.
function keyFrom(headText) {
  return crypto.createHash('sha1').update(String(headText || '')).digest('hex').slice(0, 16);
}

function readStore() {
  try {
    return JSON.parse(fs.readFileSync(STORE_PATH, 'utf8')) || {};
  } catch { return {}; }
}

// Atomic tmp+rename — never write the store in place (matches engine-flag.js).
function writeStore(store) {
  const tmp = STORE_PATH + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(store));
  fs.renameSync(tmp, STORE_PATH);
}

// applyDamp(tier, key, now) → { tier, damped, prevTier }
// Looks up the previous effective tier for `key`, clamps the downgrade, persists
// the *effective* tier as the new prev (so descent walks one step per prompt),
// and prunes entries older than the TTL. Any error → raw tier, damped:false.
function applyDamp(tier, key, now) {
  if (process.env.CALIBRA_DAMP_DISABLED) return { tier, damped: false, prevTier: null };
  try {
    const store = readStore();
    const entry = store[key];
    const fresh = entry && (now - entry.ts) <= TTL_MS;
    const prevTier = fresh ? entry.tier : null;

    const eff = prevTier ? damp(tier, prevTier) : tier;

    // Prune stale entries, then record this conversation's effective tier.
    for (const k of Object.keys(store)) {
      if (!store[k] || (now - store[k].ts) > TTL_MS) delete store[k];
    }
    store[key] = { tier: eff, ts: now };
    writeStore(store);

    return { tier: eff, damped: eff !== tier, prevTier };
  } catch {
    return { tier, damped: false, prevTier: null };
  }
}

module.exports = { damp, applyDamp, keyFrom, ORDER, NAMES, MID, STORE_PATH, TTL_MS };
