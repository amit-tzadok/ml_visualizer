# API Documentation

## Questions Database

This API provides endpoints for storing and retrieving user questions from the ML Assistant.

### Setup

1. Install dependencies:
```bash
npm install better-sqlite3
```

2. The database will be automatically created at `api/questions.db` on first use.

### Database Schema

The `questions` table stores:
- `id` - Auto-incrementing primary key
- `question` - The user's question text
- `classifier` - Current classifier type (linear, poly, mlp, knn)
- `compare_mode` - Whether compare mode was enabled
- `speed_scale` - Current speed scale value
- `timestamp` - When the question was asked
- `session_id` - Unique session identifier
- `user_agent` - User's browser (truncated for privacy)
- `ip_hash` - Hashed IP address (for privacy - not reversible)

### API Endpoints

#### POST /api/questions

Save a new question to the database.

**Request Body:**
```json
{
  "question": "What is a linear perceptron?",
  "classifier": "linear",
  "compareMode": false,
  "speedScale": 1.5,
  "sessionId": "session_12345_abc"
}
```

**Response:**
```json
{
  "success": true,
  "id": 123,
  "timestamp": "2025-11-12T10:30:00.000Z"
}
```

#### GET /api/questions

Retrieve stored questions.

**Query Parameters:**
- `limit` (optional) - Max number of questions to return (default: 50, max: 100)
- `offset` (optional) - Number of questions to skip (default: 0)
- `sessionId` (optional) - Filter by session ID
- `stats` (optional) - Set to "true" to get statistics instead of questions

**Examples:**

Get recent questions:
```
GET /api/questions?limit=10
```

Get questions for a specific session:
```
GET /api/questions?sessionId=session_12345_abc
```

Get statistics:
```
GET /api/questions?stats=true
```

**Response (questions):**
```json
{
  "success": true,
  "count": 10,
  "limit": 10,
  "offset": 0,
  "questions": [
    {
      "id": 123,
      "question": "What is a linear perceptron?",
      "classifier": "linear",
      "compare_mode": 0,
      "speed_scale": 1.5,
      "timestamp": "2025-11-12T10:30:00.000Z",
      "session_id": "session_12345_abc"
    }
  ]
}
```

**Response (stats):**
```json
{
  "success": true,
  "stats": {
    "total": 1250,
    "byClassifier": {
      "linear": 450,
      "poly": 320,
      "mlp": 280,
      "knn": 200
    },
    "last24Hours": 45
  }
}
```

### Privacy & Security

- **IP addresses** are hashed using SHA-256 and truncated. The original IP cannot be recovered.
- **User agents** are truncated to 200 characters.
- **No personal information** is collected beyond what's needed for analytics.
- Questions are stored as-is, so users should avoid including sensitive information.

### Local Development

The database file (`questions.db`) is created automatically in the `api/` directory. To reset the database, simply delete this file.

### Deployment

When deploying to Vercel or similar serverless platforms:
1. Ensure `better-sqlite3` is installed
2. The database will be created in the `/tmp` directory in serverless environments
3. Consider using a persistent database like PostgreSQL for production use

### Example Usage in Frontend

```typescript
// Save a question
await fetch('/api/questions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: userInput,
    classifier: currentClassifier,
    compareMode: isCompareMode,
    speedScale: currentSpeed,
    sessionId: sessionId
  })
});

// Get recent questions
const response = await fetch('/api/questions?limit=20');
const data = await response.json();
console.log(data.questions);
```

### Database Utilities

The `db.js` module provides helper functions:

- `initDB()` - Initialize database and create tables
- `saveQuestion(data)` - Save a question
- `getQuestions(limit, offset)` - Get recent questions
- `getQuestionsBySession(sessionId)` - Get questions for a session
- `getStats()` - Get database statistics
- `closeDB()` - Close database connection

These functions are used internally by the API endpoint but can also be used directly in other Node.js code.
