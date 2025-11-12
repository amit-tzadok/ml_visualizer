#!/usr/bin/env node

/**
 * Clear or delete the questions database.
 * Usage:
 *   node scripts/clear-db.cjs           # clears all rows (DELETE FROM questions; VACUUM;)
 *   node scripts/clear-db.cjs --drop    # delete the database file entirely
 *   node scripts/clear-db.cjs --drop --yes  # delete without confirmation
 */

const path = require('path');
const fs = require('fs');
const db = require(path.join(__dirname, '..', 'api', 'db.cjs'));

const args = process.argv.slice(2);
const doDrop = args.includes('--drop');
const force = args.includes('--yes') || args.includes('-y');

if (doDrop) {
  if (!force) {
    console.log('WARNING: This will permanently delete the database file at:', path.join(__dirname, '..', 'api', 'questions.db'));
    console.log('Run with --yes to confirm.');
    process.exit(1);
  }

  const ok = db.deleteDatabaseFile();
  if (ok) {
    console.log('✅ Database file deleted.');
    process.exit(0);
  } else {
    console.error('❌ Failed to delete database file.');
    process.exit(2);
  }
} else {
  // clear rows
  try {
    const deleted = db.clearQuestions();
    console.log(`✅ Cleared questions table. Rows deleted: ${deleted}`);
    process.exit(0);
  } catch (err) {
    console.error('❌ Failed to clear questions table:', err.message || err);
    process.exit(2);
  }
}
