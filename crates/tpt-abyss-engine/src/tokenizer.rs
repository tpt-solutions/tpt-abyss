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

    /// Resolve the model's end-of-sequence token id, trying common special
    /// token strings. Returns `None` if none are present in the vocabulary.
    pub fn eos_token_id(&self) -> Option<u32> {
        for sym in ["<|im_end|>", "<|endoftext|>", "</s>", "<|end_of_text|>"] {
            if let Some(id) = self.inner.token_to_id(sym) {
                return Some(id);
            }
        }
        None
    }
}
