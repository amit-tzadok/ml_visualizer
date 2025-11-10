export async function getOpenAIEmbedding(_: string[], __?: string): Promise<number[][]> {
  // Embeddings support removed by request — this project is configured to not use any
  // third-party embedding APIs or store API keys. If you need embeddings later, re-enable
  // a proxy or provide credentials and restore the implementation.
  throw new Error(
    'Embeddings disabled: this project is configured not to use OpenAI or other external embedding APIs.'
  );
}
