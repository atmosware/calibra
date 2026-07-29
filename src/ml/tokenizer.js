'use strict';

const fs   = require('fs');
const os   = require('os');
const path = require('path');

// vocab.txt: bert-base-uncased (same vocab as all-MiniLM-L6-v2).
// Runtime: ~/.claude-corp/calibra/ml/vocab.txt (installed by install.js or download)
// Dev:     src/ml/vocab.txt (bundled with package)
const VOCAB_PATH =
  process.env.CALIBRA_ML_VOCAB_PATH ||
  path.join(os.homedir(), '.claude-corp', 'calibra', 'ml', 'vocab.txt');

let _vocab       = null; // Map<token, id>
let _vocabLoaded = false;

function loadVocab() {
  if (_vocabLoaded) return _vocab;
  _vocabLoaded = true;
  try {
    const lines = fs.readFileSync(VOCAB_PATH, 'utf8').split('\n');
    _vocab = new Map();
    lines.forEach((line, i) => {
      const tok = line.replace(/\r$/, '');
      if (tok !== '') _vocab.set(tok, i);
    });
    return _vocab;
  } catch {
    _vocab = null;
    return null;
  }
}

// ── Basic tokenization (bert-base-uncased) ────────────────────────────────────

function isPunct(ch) {
  const cp = ch.codePointAt(0);
  return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
         (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) ||
         /\p{P}/u.test(ch);
}

function isChinese(cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF)  ||
         (cp >= 0x3400 && cp <= 0x4DBF)  ||
         (cp >= 0x20000 && cp <= 0x2A6DF) ||
         (cp >= 0xF900 && cp <= 0xFAFF);
}

function basicTokenize(text) {
  // lowercase + NFD accent-strip
  text = text.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');

  const tokens = [];
  let cur = '';

  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (/\s/.test(ch)) {
      if (cur) { tokens.push(cur); cur = ''; }
    } else if (isPunct(ch) || isChinese(cp)) {
      if (cur) { tokens.push(cur); cur = ''; }
      tokens.push(ch);
    } else {
      cur += ch;
    }
  }
  if (cur) tokens.push(cur);
  return tokens;
}

// ── WordPiece tokenization ────────────────────────────────────────────────────

const UNK_TOKEN  = '[UNK]';
const MAX_CHARS  = 100;

function wordpieceTokenize(word, vocab) {
  if (word.length > MAX_CHARS) return [UNK_TOKEN];
  if (vocab.has(word)) return [word];

  const subTokens = [];
  let start = 0;
  let isBad  = false;

  while (start < word.length) {
    let end   = word.length;
    let found = null;
    while (start < end) {
      const sub = (start === 0 ? '' : '##') + word.slice(start, end);
      if (vocab.has(sub)) { found = sub; break; }
      end--;
    }
    if (!found) { isBad = true; break; }
    subTokens.push(found);
    start = end;
  }

  return isBad ? [UNK_TOKEN] : subTokens;
}

// ── Public API ────────────────────────────────────────────────────────────────

const CLS_TOKEN = '[CLS]';
const SEP_TOKEN = '[SEP]';

/**
 * Tokenize `text` for BERT / all-MiniLM-L6-v2 inference.
 * Returns { input_ids, attention_mask, token_type_ids } as Int32Arrays.
 * Falls back to [CLS][UNK][SEP] if vocab is unavailable.
 * Tail-truncates at maxLength — use tokenizeChunks() for long-prompt inference.
 *
 * @param {string} text
 * @param {number} [maxLength=256]
 */
function tokenize(text, maxLength = 256) {
  const vocab = loadVocab();

  if (!vocab) {
    return {
      input_ids:      new Int32Array([101, 100, 102]),
      attention_mask: new Int32Array([1,   1,   1  ]),
      token_type_ids: new Int32Array([0,   0,   0  ]),
    };
  }

  const clsId = vocab.get(CLS_TOKEN) ?? 101;
  const sepId = vocab.get(SEP_TOKEN) ?? 102;
  const maxBody = maxLength - 2;
  const ids = encodeWords(text, vocab).slice(0, maxBody);

  return wrapChunk(ids, clsId, sepId);
}

function encodeWords(text, vocab) {
  const unkId = vocab.get(UNK_TOKEN) ?? 100;
  const words = basicTokenize(text || '');
  const ids   = [];
  for (const word of words) {
    const subs = wordpieceTokenize(word, vocab);
    for (const sw of subs) ids.push(vocab.get(sw) ?? unkId);
  }
  return ids;
}

function wrapChunk(bodyIds, clsId, sepId) {
  const full = [clsId, ...bodyIds, sepId];
  const len  = full.length;
  return {
    input_ids:      new Int32Array(full),
    attention_mask: new Int32Array(len).fill(1),
    token_type_ids: new Int32Array(len).fill(0),
  };
}

/**
 * Split `text` into overlapping windows of up to `maxLength` tokens each, so a
 * long prompt gets pooled across chunks instead of silently tail-truncated
 * (MiniLM's own trained max sequence length is 256 — this works within that,
 * not around it). When the prompt overflows `maxChunks` windows, keeps the
 * head and tail windows and drops the middle: the task is usually stated up
 * front and constraints/edge-cases at the end, with elaboration in between.
 *
 * @param {string} text
 * @param {number} [maxLength=256]
 * @param {{stride?: number, maxChunks?: number}} [opts]
 * @returns {Array<{input_ids: Int32Array, attention_mask: Int32Array, token_type_ids: Int32Array}>}
 */
function tokenizeChunks(text, maxLength = 256, opts = {}) {
  const stride    = opts.stride ?? 56;
  const maxChunks = opts.maxChunks ?? 4;

  const vocab = loadVocab();
  if (!vocab) {
    return [{
      input_ids:      new Int32Array([101, 100, 102]),
      attention_mask: new Int32Array([1,   1,   1  ]),
      token_type_ids: new Int32Array([0,   0,   0  ]),
    }];
  }

  const clsId   = vocab.get(CLS_TOKEN) ?? 101;
  const sepId   = vocab.get(SEP_TOKEN) ?? 102;
  const maxBody = maxLength - 2;

  const ids = encodeWords(text, vocab);
  if (ids.length <= maxBody) return [wrapChunk(ids, clsId, sepId)];

  const step = Math.max(1, maxBody - stride);
  const windows = [];
  for (let start = 0; start < ids.length; start += step) {
    const end = Math.min(start + maxBody, ids.length);
    windows.push(ids.slice(start, end));
    if (end >= ids.length) break;
  }

  let picked = windows;
  if (windows.length > maxChunks) {
    const head = Math.ceil(maxChunks / 2);
    const tail = maxChunks - head;
    picked = [...windows.slice(0, head), ...windows.slice(windows.length - tail)];
  }

  return picked.map(w => wrapChunk(w, clsId, sepId));
}

module.exports = { tokenize, tokenizeChunks, loadVocab, VOCAB_PATH };
