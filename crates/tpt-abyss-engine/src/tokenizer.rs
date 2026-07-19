//! Thin tokenizer wrapper around the `tokenizers` crate, used by the CLI to
//! turn prompts into token ids and decode generated tokens back to text.

use tokenizers::Tokenizer as HfTokenizer;

/// Wrapper over a HuggingFace `tokenizers` tokenizer JSON file.
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` path.
    pub fn from_file(path: &str) -> Result<Self, tpt_abyss_types::AbyssError> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| tpt_abyss_types::AbyssError::Engine(format!("tokenizer load: {e}")))?;
        Ok(Self { inner })
    }

    /// Encode text into token ids (no special tokens by default).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, tpt_abyss_types::AbyssError> {
        let ids = self
            .inner
            .encode(text, false)
            .map_err(|e| tpt_abyss_types::AbyssError::Engine(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        Ok(ids)
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, tpt_abyss_types::AbyssError> {
        let text = self
            .inner
            .decode(ids, false)
            .map_err(|e| tpt_abyss_types::AbyssError::Engine(format!("decode: {e}")))?;
        Ok(text)
    }
}
