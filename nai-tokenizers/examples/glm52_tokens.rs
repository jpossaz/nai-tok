//! Build a realistic GLM-5.2 prompt, tokenize it with the embedded GLM-5.2
//! tokenizer, and emit the token IDs as a JSON array on stdout (the prompt
//! string + token count go to stderr). Pipe stdout into a vLLM /v1/completions
//! call to verify the full template->tokenize->model path end to end.
//!
//!   cargo run --example glm52_tokens \
//!     --no-default-features --features "glm45_template,glm52_tokenizer,native"

use nai_tokenizers::glm45_template::{
    Chat, ContextState, Message, PrefillType, ReasoningEnabled, Version,
};
use nai_tokenizers::glm52_tokenizer::{tokenize, SpecialTokens};

fn main() {
    let chat = Chat {
        messages: vec![
            Message::System {
                content: "You are the narrator of a dark-fantasy visual novel. Reply in two vivid sentences.".to_string(),
            },
            Message::User {
                content: "The hero steps into the cursed forest.".to_string(),
            },
        ],
    };

    let prompt = ContextState::new_with_version(ReasoningEnabled::Yes, Version::GLM52)
        .chat(&chat, PrefillType::Canonical);

    let ids = tokenize(&prompt, SpecialTokens::Keep).expect("tokenization failed");

    eprintln!("--- prompt string ---\n{}\n--- {} tokens ---", prompt, ids.len());
    println!("{}", serde_json::to_string(&ids).unwrap());
}
