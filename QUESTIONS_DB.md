# Questions Database Implementation

This document describes the database system added to store user questions from the ML Assistant.

## Overview

A SQLite database has been integrated to store all user questions asked to the ML Assistant, along with contextual information like the current classifier, speed settings, and session details.

## What Was Added

### 1. Database Module (`api/db.js`)
- Initializes SQLite database with `better-sqlite3`
- Creates `questions` table with appropriate schema
- Provides utility functions:
  - `saveQuestion()` - Save a user question
  - `getQuestions()` - Retrieve recent questions
  - `getQuestionsBySession()` - Get questions for a specific session
  - `getStats()` - Get database statistics
  - `closeDB()` - Clean shutdown

### 2. API Endpoint (`api/questions.js`)
- **POST /api/questions** - Save new questions
- **GET /api/questions** - Retrieve questions with filtering
- Supports CORS for local development
- Privacy-focused: hashes IP addresses, truncates user agents
- Validates input and prevents abuse

### 3. Frontend Integration (`src/components/AgentPanel.tsx`)
- Generates unique session ID for each chat session
- Automatically saves every question to database via API
- Non-blocking: doesn't disrupt user experience if save fails
- Passes contextual information (classifier, compare mode, speed)

### 4. Configuration & Documentation
- Updated `.env.example` with database settings
- Added `api/README.md` with complete API documentation
- Created `scripts/view-db.js` for easy database inspection
- Updated `.gitignore` to exclude database files
- Added `better-sqlite3` to `package.json`

## Database Schema

```sql
CREATE TABLE questions (
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
```

Indexes created on:
- `timestamp` (for fast recent queries)
- `classifier` (for analytics by algorithm)
- `session_id` (for session-based queries)

## Privacy & Security

✅ **IP addresses are hashed** - Original IPs cannot be recovered  
✅ **User agents truncated** to 200 characters  
✅ **No personal information** collected  
✅ **Input validation** - Max 5000 chars per question  
✅ **Silent failures** - Database errors don't affect UX  

## Installation

```bash
# Install the SQLite dependency
npm install better-sqlite3

# Database is created automatically on first use
# Located at: api/questions.db
```

## Usage

### Viewing Database Contents

```bash
# View recent questions
node scripts/view-db.js

# View statistics
node scripts/view-db.js --stats

# View all questions
node scripts/view-db.js --all
```

### API Examples

```bash
# Get recent questions
curl http://localhost:5173/api/questions?limit=10

# Get statistics
curl http://localhost:5173/api/questions?stats=true

# Save a question
curl -X POST http://localhost:5173/api/questions \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a perceptron?",
    "classifier": "linear",
    "compareMode": false,
    "speedScale": 1.5,
    "sessionId": "session_123"
  }'
```

## Data Analysis Ideas

With this database, you can now:

1. **Track popular questions** - See what users ask most
2. **Analyze by classifier** - Which algorithms generate more questions?
3. **Session analysis** - How many questions per user session?
4. **Time-based trends** - When are users most active?
5. **Improve documentation** - Focus on frequently asked topics
6. **A/B testing** - Compare different UI/UX approaches

## File Structure

```
api/
├── db.js              # Database utilities
├── questions.js       # API endpoint
├── questions.db       # SQLite database (auto-created, gitignored)
└── README.md          # API documentation

scripts/
└── view-db.js         # Database viewer utility

src/components/
└── AgentPanel.tsx     # Updated with DB integration
```

## Next Steps

Consider these enhancements:

- [ ] Add admin dashboard to view analytics
- [ ] Export questions to CSV for analysis
- [ ] Implement question search/filtering UI
- [ ] Add database backup script
- [ ] Migrate to PostgreSQL for production
- [ ] Add response tracking (not just questions)
- [ ] Implement rate limiting per session
- [ ] Add data retention policies (auto-delete old data)

## Testing

The database integration:
- ✅ Doesn't block UI operations
- ✅ Fails silently if API is unavailable
- ✅ Works with existing agent functionality
- ✅ Preserves user privacy
- ✅ Provides useful analytics data

## Deployment Notes

### Local Development
- Database file: `api/questions.db`
- Persists across server restarts
- Can be deleted to reset

### Vercel/Serverless
- In serverless environments, SQLite databases are ephemeral
- Consider using Vercel Postgres or another persistent database
- Current implementation will work but data won't persist between deployments

### Alternative Databases
To switch to PostgreSQL or another database:
1. Replace `better-sqlite3` with appropriate driver
2. Update `api/db.js` with new connection logic
3. Keep the same API interface
