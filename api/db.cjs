// Database utility for storing user questions
// Uses SQLite via better-sqlite3 for simple, serverless storage

const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

// Database file location - stores in api directory
const DB_PATH = path.join(__dirname, 'questions.db');
const DB_DIR = path.dirname(DB_PATH);

// Ensure directory exists
if (!fs.existsSync(DB_DIR)) {
  fs.mkdirSync(DB_DIR, { recursive: true });
}

let db;

/**
 * Initialize the database and create tables if they don't exist
 */
function initDB() {
  if (db) return db;
  
  db = new Database(DB_PATH);
  
  // Enable WAL mode for better concurrent access
  db.pragma('journal_mode = WAL');
  
  // Create questions table
  db.exec(`
    CREATE TABLE IF NOT EXISTS questions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      question TEXT NOT NULL,
      classifier TEXT,
      compare_mode BOOLEAN DEFAULT 0,
      speed_scale REAL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      session_id TEXT,
      user_agent TEXT,
      ip_hash TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_timestamp ON questions(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_classifier ON questions(classifier);
    CREATE INDEX IF NOT EXISTS idx_session ON questions(session_id);
  `);
  
  return db;
}

/**
 * Save a user question to the database
 * @param {Object} data - Question data
 * @param {string} data.question - The user's question text
 * @param {string} data.classifier - Current classifier type
 * @param {boolean} data.compareMode - Whether compare mode is enabled
 * @param {number} data.speedScale - Current speed scale
 * @param {string} data.sessionId - Optional session identifier
 * @param {string} data.userAgent - Optional user agent string
 * @param {string} data.ipHash - Optional hashed IP for privacy
 * @returns {Object} Inserted row data
 */
function saveQuestion(data) {
  const db = initDB();
  
  const stmt = db.prepare(`
    INSERT INTO questions (
      question, classifier, compare_mode, speed_scale, 
      session_id, user_agent, ip_hash
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
  `);
  
  const info = stmt.run(
    data.question,
    data.classifier || null,
    data.compareMode ? 1 : 0,
    data.speedScale || null,
    data.sessionId || null,
    data.userAgent || null,
    data.ipHash || null
  );
  
  return {
    id: info.lastInsertRowid,
    ...data,
    timestamp: new Date().toISOString()
  };
}

/**
 * Get recent questions
 * @param {number} limit - Maximum number of questions to retrieve
 * @param {number} offset - Number of questions to skip
 * @returns {Array} Array of question objects
 */
function getQuestions(limit = 50, offset = 0) {
  const db = initDB();
  
  const stmt = db.prepare(`
    SELECT 
      id, question, classifier, compare_mode, speed_scale,
      timestamp, session_id
    FROM questions
    ORDER BY timestamp DESC
    LIMIT ? OFFSET ?
  `);
  
  return stmt.all(limit, offset);
}

/**
 * Get questions for a specific session
 * @param {string} sessionId - Session identifier
 * @returns {Array} Array of question objects
 */
function getQuestionsBySession(sessionId) {
  const db = initDB();
  
  const stmt = db.prepare(`
    SELECT 
      id, question, classifier, compare_mode, speed_scale, timestamp
    FROM questions
    WHERE session_id = ?
    ORDER BY timestamp ASC
  `);
  
  return stmt.all(sessionId);
}

/**
 * Get question statistics
 * @returns {Object} Statistics about stored questions
 */
function getStats() {
  const db = initDB();
  
  const total = db.prepare('SELECT COUNT(*) as count FROM questions').get();
  const byClassifier = db.prepare(`
    SELECT classifier, COUNT(*) as count 
    FROM questions 
    WHERE classifier IS NOT NULL
    GROUP BY classifier
  `).all();
  
  const recentCount = db.prepare(`
    SELECT COUNT(*) as count 
    FROM questions 
    WHERE timestamp > datetime('now', '-24 hours')
  `).get();
  
  return {
    total: total.count,
    byClassifier: byClassifier.reduce((acc, row) => {
      acc[row.classifier] = row.count;
      return acc;
    }, {}),
    last24Hours: recentCount.count
  };
}

/**
 * Close the database connection
 */
function closeDB() {
  if (db) {
    db.close();
    db = null;
  }
}

/**
 * Delete all rows from the questions table
 * @returns {number} number of rows deleted
 */
function clearQuestions() {
  const db = initDB();
  const info = db.prepare('DELETE FROM questions').run();
  // Reclaim space
  try {
    db.prepare('VACUUM').run();
  } catch (e) {
    // ignore vacuum errors
  }
  return info.changes || 0;
}

/**
 * Delete the database file entirely. Closes DB first.
 * @returns {boolean} true if file was removed or didn't exist
 */
function deleteDatabaseFile() {
  try {
    closeDB();
  } catch (e) {
    // ignore
  }
  try {
    if (fs.existsSync(DB_PATH)) {
      fs.unlinkSync(DB_PATH);
    }
    return true;
  } catch (err) {
    return false;
  }
}

// Clean shutdown on process exit
process.on('exit', closeDB);
process.on('SIGINT', () => {
  closeDB();
  process.exit(0);
});

module.exports = {
  initDB,
  saveQuestion,
  getQuestions,
  getQuestionsBySession,
  getStats,
  closeDB,
  clearQuestions,
  deleteDatabaseFile
};
