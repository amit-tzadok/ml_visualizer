#!/usr/bin/env node

/**
 * Database viewer utility
 * View and query the questions database from the command line
 * 
 * Usage:
 *   node scripts/view-db.js           # Show recent questions
 *   node scripts/view-db.js --stats   # Show statistics
 *   node scripts/view-db.js --all     # Show all questions
 */

const path = require('path');
const fs = require('fs');

// Load the database utilities
const dbPath = path.join(__dirname, '..', 'api', 'db.cjs');
const db = require(dbPath);

const args = process.argv.slice(2);
const showStats = args.includes('--stats') || args.includes('-s');
const showAll = args.includes('--all') || args.includes('-a');
const limit = showAll ? 10000 : 20;

console.log('📊 ML Visualizer - Questions Database Viewer\n');

// Check if database exists
const dbFilePath = path.join(__dirname, '..', 'api', 'questions.db');
if (!fs.existsSync(dbFilePath)) {
  console.log('⚠️  Database file not found. No questions have been saved yet.');
  console.log(`   Expected location: ${dbFilePath}\n`);
  process.exit(0);
}

try {
  if (showStats) {
    // Show statistics
    console.log('📈 Database Statistics:\n');
    const stats = db.getStats();
    
    console.log(`Total Questions: ${stats.total}`);
    console.log(`Last 24 Hours: ${stats.last24Hours}`);
    console.log('\nBy Classifier:');
    
    Object.entries(stats.byClassifier).forEach(([classifier, count]) => {
      const bar = '█'.repeat(Math.floor(count / 10));
      console.log(`  ${classifier.padEnd(8)} ${String(count).padStart(4)} ${bar}`);
    });
    
  } else {
    // Show questions
    console.log(`📝 Recent Questions (showing ${limit}):\n`);
    const questions = db.getQuestions(limit, 0);
    
    if (questions.length === 0) {
      console.log('No questions found in the database.');
    } else {
      questions.forEach((q, idx) => {
        console.log(`[${idx + 1}] ${new Date(q.timestamp).toLocaleString()}`);
        console.log(`    Classifier: ${q.classifier || 'N/A'} | Speed: ${q.speed_scale || 'N/A'}x | Compare: ${q.compare_mode ? 'Yes' : 'No'}`);
        console.log(`    Q: ${q.question}`);
        if (q.session_id) {
          console.log(`    Session: ${q.session_id}`);
        }
        console.log('');
      });
    }
  }
  
  console.log(`\n💾 Database: ${dbFilePath}`);
  console.log(`📦 Size: ${(fs.statSync(dbFilePath).size / 1024).toFixed(2)} KB\n`);
  
} catch (error) {
  console.error('❌ Error reading database:', error.message);
  process.exit(1);
} finally {
  db.closeDB();
}
