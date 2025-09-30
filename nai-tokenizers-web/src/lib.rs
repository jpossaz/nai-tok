use nai_tokenizers::glm45_tokenizer::{self, SpecialTokens};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

#[derive(Serialize, Deserialize)]
pub struct TokenInfo {
    pub id: u32,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Serialize, Deserialize)]
pub struct TokenizeResult {
    pub tokens: Vec<TokenInfo>,
    pub ids: Vec<u32>,
}

/// Tokenizes the input text and returns detailed information about each token
#[wasm_bindgen]
pub fn tokenize(input: &str, keep_special_tokens: bool) -> Result<JsValue, JsValue> {
    console_log!("Tokenizing input of length: {}", input.len());

    let special_tokens = if keep_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };

    // Get token IDs
    let ids = glm45_tokenizer::tokenize(input, special_tokens)
        .map_err(|e| JsValue::from_str(&format!("Tokenization error: {}", e)))?;

    console_log!("Generated {} tokens", ids.len());

    // Decode each token individually to get the text
    let mut tokens = Vec::new();
    let mut current_pos = 0;

    for (_idx, &id) in ids.iter().enumerate() {
        // Decode this single token to get its text
        let token_text = glm45_tokenizer::detokenize(&[id], special_tokens)
            .map_err(|e| JsValue::from_str(&format!("Detokenization error: {}", e)))?;

        let token_len = token_text.len();

        tokens.push(TokenInfo {
            id,
            text: token_text,
            start: current_pos,
            end: current_pos + token_len,
        });

        current_pos += token_len;
    }

    let result = TokenizeResult {
        tokens,
        ids: ids.clone(),
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Decodes token IDs back to text
#[wasm_bindgen]
pub fn detokenize(ids: Vec<u32>, keep_special_tokens: bool) -> Result<String, JsValue> {
    console_log!("Detokenizing {} tokens", ids.len());

    let special_tokens = if keep_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };

    glm45_tokenizer::detokenize(&ids, special_tokens)
        .map_err(|e| JsValue::from_str(&format!("Detokenization error: {}", e)))
}

/// Returns information about the tokenizer
#[wasm_bindgen]
pub fn get_tokenizer_info() -> JsValue {
    let info = serde_json::json!({
        "name": "GLM-4.5 Tokenizer",
        "version": "0.1.0",
        "description": "NovelAI GLM-4.5 tokenizer for WebAssembly"
    });

    serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL)
}