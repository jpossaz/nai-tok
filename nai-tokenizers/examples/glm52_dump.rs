//! Dump GLM-4.x vs GLM-5.2 template output for eyeballing against the real
//! chat templates. Run with:
//!   cargo run --example glm52_dump --no-default-features --features glm45_template

use nai_tokenizers::glm45_template::{
    Chat, ContextState, Message, PrefillType, ReasoningEnabled, Version,
};

fn sample_chat() -> Chat {
    Chat {
        messages: vec![
            Message::System {
                content: "You are a game master.".to_string(),
            },
            Message::User {
                content: "Begin the adventure.".to_string(),
            },
        ],
    }
}

fn render(version: Version) -> String {
    ContextState::new_with_version(ReasoningEnabled::Yes, version).chat(
        &sample_chat(),
        PrefillType::FullReasoning {
            reasoning_content: "Let me plan.".to_string(),
            content: "".to_string(),
        },
    )
}

fn main() {
    println!("===== GLM456 (FullReasoning prefill) =====");
    println!("{}", render(Version::GLM456));
    println!("\n===== GLM52 (FullReasoning prefill) =====");
    println!("{}", render(Version::GLM52));

    println!("\n===== GLM52 canonical prefill =====");
    let canonical = ContextState::new_with_version(ReasoningEnabled::Yes, Version::GLM52)
        .chat(&sample_chat(), PrefillType::Canonical);
    println!("{}", canonical);
}
