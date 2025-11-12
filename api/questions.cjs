// API endpoint for storing and retrieving user questions
// This endpoint handles both POST (save) and GET (retrieve) requests

const crypto = require('crypto');
const { saveQuestion, getQuestions, getQuestionsBySession, getStats, clearQuestions, deleteDatabaseFile } = require('./db.cjs');

/**
 * Hash IP address for privacy - we don't store actual IPs
 * @param {string} ip - IP address
 * @returns {string} Hashed IP
 */
function hashIP(ip) {
  if (!ip) return null;
  return crypto.createHash('sha256').update(ip).digest('hex').substring(0, 16);
}

/**
 * Main request handler
 */
module.exports = async (req, res) => {
  // Enable CORS for local development
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  try {
    // POST - Save a new question
    if (req.method === 'POST') {
      const body = req.body;
      
      // Validate required fields
      if (!body || !body.question || typeof body.question !== 'string') {
        return res.status(400).json({ 
          error: 'Missing or invalid question field' 
        });
      }
      
      // Validate question length (prevent abuse)
      if (body.question.length > 5000) {
        return res.status(400).json({ 
          error: 'Question too long (max 5000 characters)' 
        });
      }
      
      // Extract metadata
      const userAgent = req.headers['user-agent'] || null;
      const ip = req.headers['x-forwarded-for'] || 
                 req.headers['x-real-ip'] || 
                 req.connection?.remoteAddress || 
                 null;
      const ipHash = hashIP(ip);
      
      // Save to database
      const result = saveQuestion({
        question: body.question.trim(),
        classifier: body.classifier || null,
        compareMode: Boolean(body.compareMode),
        speedScale: typeof body.speedScale === 'number' ? body.speedScale : null,
        sessionId: body.sessionId || null,
        userAgent: userAgent ? userAgent.substring(0, 200) : null,
        ipHash
      });
      
      return res.status(201).json({
        success: true,
        id: result.id,
        timestamp: result.timestamp
      });
    }
    
    // GET - Retrieve questions
    if (req.method === 'GET') {
      const { limit, offset, sessionId, stats } = req.query || {};
      
      // Return statistics if requested
      if (stats === 'true' || stats === '1') {
        const statistics = getStats();
        return res.status(200).json({
          success: true,
          stats: statistics
        });
      }
      
      // Return questions for a specific session
      if (sessionId) {
        const questions = getQuestionsBySession(sessionId);
        return res.status(200).json({
          success: true,
          count: questions.length,
          questions
        });
      }
      
      // Return recent questions
      const limitNum = Math.min(parseInt(limit) || 50, 100); // Max 100
      const offsetNum = parseInt(offset) || 0;
      
      const questions = getQuestions(limitNum, offsetNum);
      
      return res.status(200).json({
        success: true,
        count: questions.length,
        limit: limitNum,
        offset: offsetNum,
        questions
      });
    }
    
    // DELETE - clear or drop the database
    if (req.method === 'DELETE') {
      // safety: require explicit confirm flag
      const { confirm, drop } = req.query || {};
      if (!(confirm === '1' || confirm === 'true')) {
        return res.status(400).json({ error: 'Deletion requires confirm=1 query parameter' });
      }

      if (drop === '1' || drop === 'true') {
        const ok = deleteDatabaseFile();
        if (ok) return res.status(200).json({ success: true, dropped: true });
        return res.status(500).json({ success: false, error: 'Failed to delete database file' });
      }

      // Otherwise clear all rows
      const deleted = clearQuestions();
      return res.status(200).json({ success: true, deleted });
    }

    // Method not allowed
    return res.status(405).json({ 
      error: 'Method not allowed. Use GET or POST.' 
    });
    
  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};
