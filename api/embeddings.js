// Embeddings endpoint removed.
// The project is intentionally configured to not use any third-party embedding APIs or
// store API keys. This endpoint always returns 501 Not Implemented to avoid accidental
// outbound requests or reliance on API keys.
module.exports = async (_req, res) => {
  res.status(501).json({ error: 'Embeddings endpoint disabled: project not using external APIs.' });
};
