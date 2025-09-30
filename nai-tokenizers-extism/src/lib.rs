use extism_pdk::*;
use nai_tokenizers::glm45_template::{
    Chat, ContextState, Message, PrefillType, ReasoningEnabled,
};
use nai_tokenizers::glm45_tokenizer::{SpecialTokens, tokenize as glm_tokenize};
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizeInput {
    text: String,
    include_special_tokens: bool,
}

// OpenAI-compatible message types
#[derive(Deserialize)]
struct ExternalMessage {
    role: String,
    content: String,
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl From<ExternalMessage> for Message {
    fn from(msg: ExternalMessage) -> Self {
        match msg.role.as_str() {
            "system" | "developer" => Message::System {
                content: msg.content,
            },
            "user" => Message::User {
                content: msg.content,
            },
            "assistant" => Message::Assistant {
                content: msg.content,
                reasoning_content: msg.reasoning_content,
            },
            _ => Message::User {
                content: msg.content,
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ExternalPrefillType {
    None,
    Canonical,
    PartialReasoning { reasoning_content: String },
    FullReasoning {
        reasoning_content: String,
        content: String,
    },
}

impl From<ExternalPrefillType> for PrefillType {
    fn from(prefill: ExternalPrefillType) -> Self {
        match prefill {
            ExternalPrefillType::None => PrefillType::None,
            ExternalPrefillType::Canonical => PrefillType::Canonical,
            ExternalPrefillType::PartialReasoning { reasoning_content } => {
                PrefillType::PartialReasoning { reasoning_content }
            }
            ExternalPrefillType::FullReasoning {
                reasoning_content,
                content,
            } => PrefillType::FullReasoning {
                reasoning_content,
                content,
            },
        }
    }
}

#[derive(Deserialize)]
struct ChatTemplateInput {
    messages: Vec<ExternalMessage>,
    #[serde(default)]
    reasoning_enabled: bool,
    #[serde(default)]
    prefill: Option<ExternalPrefillType>,
    #[serde(default)]
    ignore_message_position: bool,
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

#[plugin_fn]
pub fn chat_template(
    Msgpack(ChatTemplateInput {
        messages,
        reasoning_enabled,
        prefill,
        ignore_message_position,
    }): Msgpack<ChatTemplateInput>,
) -> FnResult<String> {
    let reasoning = if reasoning_enabled {
        ReasoningEnabled::Yes
    } else {
        ReasoningEnabled::No
    };

    let internal_messages: Vec<Message> = messages.into_iter().map(|m| m.into()).collect();
    let chat = Chat {
        messages: internal_messages,
    };

    let prefill_type = prefill
        .map(|p| p.into())
        .unwrap_or(PrefillType::Canonical);

    let result = ContextState::new(reasoning).chat_with_options(&chat, prefill_type, ignore_message_position);
    Ok(result)
}
