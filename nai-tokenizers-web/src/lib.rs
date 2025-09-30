use nai_tokenizers::glm45_tokenizer::{self, SpecialTokens};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Tokenizes the input text and returns token IDs
#[wasm_bindgen]
pub fn tokenize(text: &str, include_special_tokens: bool) -> Result<Vec<u32>, JsValue> {
    let special_tokens = if include_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };

    glm45_tokenizer::tokenize(text, special_tokens)
        .map_err(|e| JsValue::from_str(&format!("Tokenization error: {}", e)))
}

/// Decodes token IDs back to text
#[wasm_bindgen]
pub fn detokenize(ids: Vec<u32>, include_special_tokens: bool) -> Result<String, JsValue> {
    let special_tokens = if include_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };

    glm45_tokenizer::detokenize(&ids, special_tokens)
        .map_err(|e| JsValue::from_str(&format!("Detokenization error: {}", e)))
}

/// Decodes a single token ID to its text representation
#[wasm_bindgen]
pub fn decode_token(id: u32, include_special_tokens: bool) -> Result<String, JsValue> {
    let special_tokens = if include_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };

    glm45_tokenizer::detokenize(&[id], special_tokens)
        .map_err(|e| JsValue::from_str(&format!("Detokenization error: {}", e)))
}