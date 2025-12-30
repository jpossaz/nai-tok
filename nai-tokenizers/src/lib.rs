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

    pub fn vocab_size() -> usize {
        GLM45_TOKENIZER.get_vocab_size(true)
    }
}

#[cfg(feature = "glm45_template")]
pub mod glm45_template {
    use serde::Deserialize;

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

    pub enum MessagePosition {
        Intermediate,
        Last,
    }

    /// GLM 4.7 uses "</think>" instead of "<think></think>" when a reasoning is empty.
    /// It also doesn't use /nothink
    pub enum Version {
        GLM456,
        GLM47,
    }

    #[derive(Deserialize)]
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

    pub struct ContextState {
        buffer: String,
        reasoning_enabled: ReasoningEnabled,
        remove_reasoning: RemoveReasoning,
        version: Version,
    }

    pub struct Chat {
        pub messages: Vec<Message>,
    }

    pub enum PrefillType {
        None,
        Canonical,
        PartialReasoning {
            reasoning_content: String,
        },
        FullReasoning {
            reasoning_content: String,
            content: String,
        },
    }

    impl ContextState {
        pub fn new(reasoning_enabled: ReasoningEnabled) -> Self {
            Self {
                buffer: "[gMASK]<sop>".to_string(),
                reasoning_enabled: reasoning_enabled,
                remove_reasoning: RemoveReasoning::No,
                version: Version::GLM456,
            }
        }
        pub fn new_with_version(reasoning_enabled: ReasoningEnabled, version: Version) -> Self {
            Self {
                buffer: "[gMASK]<sop>".to_string(),
                reasoning_enabled: reasoning_enabled,
                remove_reasoning: RemoveReasoning::No,
                version,
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
            if matches!(self.version, Version::GLM57) {
                return self;
            }
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
            if matches!(self.version, Version::GLM57) && content.is_empty() {
                self = self.think_end();
                return self;
            }
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
        pub fn message(mut self, message: &Message, position: &MessagePosition) -> Self {
            match message {
                Message::Assistant {
                    reasoning_content,
                    content,
                } => {
                    self = self.assistant_sentinel();

                    // Cleaner reasoning logic
                    let should_include_reasoning = matches!(
                        (&self.remove_reasoning, &self.reasoning_enabled, position),
                        (
                            RemoveReasoning::No,
                            ReasoningEnabled::Yes,
                            MessagePosition::Last
                        )
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

        // QoL methods for inline message creation
        pub fn intermediate_system_message(self, content: impl Into<String>) -> Self {
            self.message(
                &Message::System {
                    content: content.into(),
                },
                &MessagePosition::Intermediate,
            )
        }

        pub fn intermediate_user_message(self, content: impl Into<String>) -> Self {
            self.message(
                &Message::User {
                    content: content.into(),
                },
                &MessagePosition::Intermediate,
            )
        }

        pub fn intermediate_assistant_message(self, content: impl Into<String>) -> Self {
            self.message(
                &Message::Assistant {
                    content: content.into(),
                    reasoning_content: None,
                },
                &MessagePosition::Intermediate,
            )
        }

        pub fn assistant_with_reasoning(
            self,
            reasoning: impl Into<String>,
            content: impl Into<String>,
        ) -> Self {
            self.message(
                &Message::Assistant {
                    reasoning_content: Some(reasoning.into()),
                    content: content.into(),
                },
                &MessagePosition::Intermediate,
            )
        }

        pub fn canonical_prefill(self) -> Self {
            if matches!(
                (&self.reasoning_enabled, &self.remove_reasoning),
                (ReasoningEnabled::Yes, RemoveReasoning::No)
            ) {
                self.assistant_sentinel()
            } else {
                self.intermediate_assistant_message("")
            }
        }

        pub fn chat(self, chat: &Chat, prefill: PrefillType) -> String {
            self.chat_with_options(chat, prefill, false)
        }

        pub fn chat_with_options(
            mut self,
            chat: &Chat,
            prefill: PrefillType,
            ignore_message_position: bool,
        ) -> String {
            for (i, message) in chat.messages.iter().enumerate() {
                let message_position = if ignore_message_position {
                    MessagePosition::Last
                } else if i == chat.messages.len() - 1 {
                    if matches!(prefill, PrefillType::None) {
                        MessagePosition::Intermediate
                    } else {
                        MessagePosition::Last
                    }
                } else {
                    MessagePosition::Intermediate
                };
                self = self.message(message, &message_position);
            }
            match prefill {
                PrefillType::None => self,
                PrefillType::Canonical => self.canonical_prefill(),
                PrefillType::PartialReasoning { reasoning_content } => self
                    .assistant_sentinel()
                    .think_start()
                    .text(&reasoning_content),
                PrefillType::FullReasoning {
                    reasoning_content,
                    content,
                } => self
                    .assistant_sentinel()
                    .thinking_content(&reasoning_content)
                    .text(&content),
            }
            .take()
        }

        pub fn take(self) -> String {
            self.buffer
        }
    }
}
