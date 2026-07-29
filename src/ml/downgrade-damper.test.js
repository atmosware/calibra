'use strict';

// Golden table for the pure clamp. No fs — applyDamp's store I/O is exercised
// separately/manually (see plan verification).
//   node src/ml/downgrade-damper.test.js

const assert = require('assert');
const { damp, keyFrom } = require('./downgrade-damper.js');

const cases = [
  // [new, prev, expected]
  // Downgrades: at most one step below prev, floored at mid.
  ['light', 'ultra', 'deep'],   // ultra → light  ⇒ deep
  ['light', 'deep',  'mid'],    // deep  → light  ⇒ mid
  ['light', 'mid',   'mid'],    // mid   → light  ⇒ mid  (floor)
  ['light', 'light', 'light'],  // light → light  ⇒ light
  ['deep',  'ultra', 'deep'],   // ultra → deep   ⇒ deep (exactly one down)
  ['mid',   'ultra', 'deep'],   // ultra → mid    ⇒ deep (only one step)
  ['mid',   'deep',  'mid'],    // deep  → mid    ⇒ mid
  // Upgrades and same-tier: never touched.
  ['ultra', 'mid',   'ultra'],  // mid   → ultra  ⇒ ultra
  ['deep',  'light', 'deep'],   // light → deep   ⇒ deep
  ['ultra', 'ultra', 'ultra'],  // same
  // Unknown tier → passthrough.
  ['light', 'bogus', 'light'],
];

let failed = 0;
for (const [nt, pt, exp] of cases) {
  const got = damp(nt, pt);
  if (got !== exp) {
    failed++;
    console.error(`FAIL damp(${nt}, ${pt}) = ${got}, expected ${exp}`);
  }
}

// keyFrom is stable and short.
assert.strictEqual(keyFrom('hello'), keyFrom('hello'), 'keyFrom must be deterministic');
assert.notStrictEqual(keyFrom('a'), keyFrom('b'), 'keyFrom must vary by input');
assert.strictEqual(keyFrom('x').length, 16, 'keyFrom must be 16 hex chars');

assert.strictEqual(failed, 0, `${failed} clamp case(s) failed`);
console.log(`downgrade-damper: ${cases.length} clamp cases + keyFrom checks passed`);
