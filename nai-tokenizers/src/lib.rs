#[cfg(feature = "glm45_tokenizer")]
pub mod glm45_tokenizer {
    use anyhow::Result;

    use tokenizers::Tokenizer;

    pub fn load() -> Result<Tokenizer> {
        // Load compressed tokenizer data
        let compressed_data = include_bytes!("../tokenizers/glm-4.5-tokenizer.json.br");

        // Decompress with Brotli
        let mut decompressed_data = Vec::new();
        brotli::BrotliDecompress(&mut &compressed_data[..], &mut decompressed_data)
            .map_err(|e| anyhow::anyhow!("Failed to decompress tokenizer: {}", e))?;

        // Parse from decompressed bytes
        let tokenizer = Tokenizer::from_bytes(&decompressed_data)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(tokenizer)
    }

    lazy_static::lazy_static! {
        pub static ref GLM45_TOKENIZER: Tokenizer = load().expect("Failed to load GLM-4.5 tokenizer");
    }

    #[derive(Clone, Copy)]
    pub enum SpecialTokens {
        Ignore,
        Keep,
    }

    impl From<SpecialTokens> for bool {
        fn from(val: SpecialTokens) -> bool {
            match val {
                SpecialTokens::Ignore => false,
                SpecialTokens::Keep => true,
            }
        }
    }

    pub fn tokenize(input: &str, special_tokens: SpecialTokens) -> Result<Vec<u32>> {
        let encoding = GLM45_TOKENIZER
            .encode(input, special_tokens.into())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn detokenize(ids: &[u32], special_tokens: SpecialTokens) -> Result<String> {
        let special_tokens: bool = special_tokens.into();
        let decoded = GLM45_TOKENIZER
            .decode(ids, !special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(decoded)
    }

    #[cfg(test)]
    mod tests {
        use std::vec;

        use super::*;
        #[test]
        fn test_tokenize() {
            let input = "[gMASK]this is a test where da goose is cooked<|system|>no<|user|>yes<|assistant|>maybe";
            let expected_output = vec![
                151331, 574, 374, 264, 1273, 1380, 2994, 61701, 374, 28998, 151335, 2152, 151336,
                9689, 151337, 36569,
            ];
            let output = tokenize(input, SpecialTokens::Keep).unwrap();
            assert_eq!(output, expected_output);
        }

        #[test]
        fn test_detokenize() {
            let input = vec![
                151331, 574, 374, 264, 1273, 1380, 2994, 61701, 374, 28998, 151335, 2152, 151336,
                9689, 151337, 36569,
            ];
            let expected_output = "[gMASK]this is a test where da goose is cooked<|system|>no<|user|>yes<|assistant|>maybe";
            let output = detokenize(&input, SpecialTokens::Keep).unwrap();
            assert_eq!(output, expected_output);
        }
    }
}

#[cfg(feature = "glm45_template")]
pub mod glm45_template {

    /// Whether to remove reasoning for the next assistant message.
    pub enum RemoveReasoning {
        No,
        Yes,
    }

    /// Whether reasoning is enabled for the model, in general.
    pub enum ReasoningEnabled {
        No,
        Yes,
    }

    pub enum GenerationPrompt {
        Add,
        Ignore,
    }

    pub enum Message {
        System {
            content: String,
        },
        User {
            content: String,
        },
        Assistant {
            content: String,
            reasoning_content: Option<String>,
        },
    }

    impl Message {
        fn content(&self) -> &str {
            match self {
                Message::System { content } => content,
                Message::User { content } => content,
                Message::Assistant { content, .. } => content,
            }
        }
    }

    struct ContextState {
        buffer: String,
        reasoning_enabled: ReasoningEnabled,
        remove_reasoning: RemoveReasoning,
    }

    impl ContextState {
        pub fn new(reasoning_enabled: ReasoningEnabled) -> Self {
            Self {
                buffer: "[gMASK]<sop>".to_string(),
                reasoning_enabled: reasoning_enabled,
                remove_reasoning: RemoveReasoning::No,
            }
        }
        pub fn system_sentinel(mut self) -> Self {
            self.buffer.push_str("<|system|>\n");
            self
        }
        pub fn user_sentinel(mut self) -> Self {
            self.buffer.push_str("<|user|>\n");
            self
        }
        pub fn assistant_sentinel(mut self) -> Self {
            self.buffer.push_str("<|assistant|>\n");
            self
        }
        pub fn nothink_sentinel(mut self) -> Self {
            self.buffer.push_str("/nothink");
            self
        }
        pub fn think_start(mut self) -> Self {
            self.buffer.push_str("<think>");
            self
        }
        pub fn think_end(mut self) -> Self {
            self.buffer.push_str("</think>\n");
            self
        }
        pub fn text(mut self, content: &str) -> Self {
            self.buffer.push_str(content);
            self
        }
        pub fn thinking_content(mut self, content: &str) -> Self {
            self = self.think_start();
            self = self.text(content);
            self = self.think_end();
            self
        }
        pub fn remove_reasoning(mut self) -> Self {
            self.remove_reasoning = RemoveReasoning::Yes;
            self
        }
        pub fn restore_reasoning(mut self) -> Self {
            self.remove_reasoning = RemoveReasoning::No;
            self
        }
        pub fn message(mut self, message: &Message) -> Self {
            match message {
                Message::Assistant {
                    reasoning_content,
                    content,
                } => {
                    self = self.assistant_sentinel();

                    // Cleaner reasoning logic
                    let should_include_reasoning = matches!(
                        (&self.remove_reasoning, &self.reasoning_enabled),
                        (RemoveReasoning::No, ReasoningEnabled::Yes)
                    );

                    if should_include_reasoning {
                        self = self.thinking_content(reasoning_content.as_deref().unwrap_or(""));
                    } else {
                        self = self.thinking_content("");
                    }

                    self = self.text(content);

                    self = self.restore_reasoning();
                }
                Message::User { content } => {
                    self = self.user_sentinel();
                    self = self.text(content);

                    // Check if nothink is already there
                    if content.ends_with("/nothink") {
                        self = self.remove_reasoning();
                    } else if matches!(self.remove_reasoning, RemoveReasoning::Yes)
                        || matches!(self.reasoning_enabled, ReasoningEnabled::No)
                    {
                        self = self.nothink_sentinel();
                        self = self.remove_reasoning();
                    }
                }
                Message::System { content } => {
                    self = self.system_sentinel();
                    self = self.text(content);
                }
            }

            self
        }
        pub fn take(self) -> String {
            self.buffer
        }
    }
}
