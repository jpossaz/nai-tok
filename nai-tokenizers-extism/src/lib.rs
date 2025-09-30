use extism_pdk::*;
use nai_tokenizers::glm45_tokenizer::{SpecialTokens, tokenize as glm_tokenize};
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizeInput {
    text: String,
    include_special_tokens: bool,
}

#[plugin_fn]
pub fn tokenize(
    Msgpack(TokenizeInput {
        text,
        include_special_tokens,
    }): Msgpack<TokenizeInput>,
) -> FnResult<Msgpack<Vec<u32>>> {
    let include_special_tokens = if include_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };
    let tokens = glm_tokenize(&text, include_special_tokens)?;
    Ok(Msgpack(tokens))
}

#[derive(Deserialize)]
struct DetokenizeInput {
    tokens: Vec<u32>,
    include_special_tokens: bool,
}

#[plugin_fn]
pub fn detokenize(
    Msgpack(DetokenizeInput {
        tokens,
        include_special_tokens,
    }): Msgpack<DetokenizeInput>,
) -> FnResult<String> {
    let include_special_tokens = if include_special_tokens {
        SpecialTokens::Keep
    } else {
        SpecialTokens::Ignore
    };
    let text = nai_tokenizers::glm45_tokenizer::detokenize(&tokens, include_special_tokens)?;
    Ok(text)
}
