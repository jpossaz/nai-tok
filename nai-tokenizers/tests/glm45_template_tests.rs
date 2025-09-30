#[cfg(feature = "glm45_template")]
mod glm45_template_tests {
    use nai_tokenizers::glm45_template::*;

    #[test]
    fn test_01_basic_with_gen_prompt() {
        let output = ContextState::new(ReasoningEnabled::No)
            .intermediate_system_message("You are a helpful assistant.")
            .intermediate_user_message("Hello, how are you?")
            .canonical_prefill()
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_02_thinking_enabled() {
        let output = ContextState::new(ReasoningEnabled::Yes)
            .intermediate_system_message("You are a helpful AI assistant.")
            .intermediate_user_message("Solve this: 2 + 2")
            .canonical_prefill()
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_03_thinking_disabled() {
        let output = ContextState::new(ReasoningEnabled::No)
            .intermediate_system_message("You are a helpful AI assistant.")
            .intermediate_user_message("Solve this: 2 + 2")
            .canonical_prefill()
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_04_multi_turn() {
        let output = ContextState::new(ReasoningEnabled::No)
            .intermediate_system_message("You are an expert programmer.")
            .intermediate_user_message("How do I read a file in Python?")
            .intermediate_assistant_message("Use `open()` with a context manager:\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```")
            .intermediate_user_message("What about writing?")
            .canonical_prefill()
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_05_no_system_prompt() {
        let output = ContextState::new(ReasoningEnabled::No)
            .intermediate_user_message("What's the weather like?")
            .intermediate_assistant_message("I don't have real-time data.")
            .intermediate_user_message("Make a guess?")
            .canonical_prefill()
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_06_reasoning_content_think_disabled() {
        let output = ContextState::new(ReasoningEnabled::No)
            .intermediate_system_message("You are a helpful math tutor.")
            .intermediate_user_message("What is 15 * 24?")
            .assistant_with_reasoning(
                "15 * 24 = 15 * (20 + 4) = 300 + 60 = 360",
                "The answer is 360.",
            )
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_07_reasoning_content_think_enabled() {
        let output = ContextState::new(ReasoningEnabled::Yes)
            .intermediate_system_message("You are a helpful math tutor.")
            .intermediate_user_message("What is 15 * 24?")
            .assistant_with_reasoning(
                "15 * 24 = 15 * (20 + 4) = 300 + 60 = 360",
                "The answer is 360.",
            )
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_08_multi_turn_reasoning() {
        let output = ContextState::new(ReasoningEnabled::Yes)
            .intermediate_user_message("Is 17 prime?")
            .assistant_with_reasoning(
                "Check divisibility up to sqrt(17) ≈ 4.12. Not divisible by 2, 3. So 17 is prime.",
                "Yes, 17 is prime.",
            )
            .intermediate_user_message("What about 18?")
            .canonical_prefill()
            .take();

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_09_chat_no_prefill() {
        let chat = Chat {
            messages: vec![
                Message::System {
                    content: "You are a helpful assistant.".to_string(),
                },
                Message::User {
                    content: "Hello!".to_string(),
                },
            ],
        };

        let output = ContextState::new(ReasoningEnabled::No).chat(&chat, PrefillType::None);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_10_chat_canonical_prefill() {
        let chat = Chat {
            messages: vec![
                Message::System {
                    content: "You are a helpful assistant.".to_string(),
                },
                Message::User {
                    content: "What is 2+2?".to_string(),
                },
            ],
        };

        let output =
            ContextState::new(ReasoningEnabled::No).chat(&chat, PrefillType::Canonical);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_11_chat_canonical_prefill_reasoning_enabled() {
        let chat = Chat {
            messages: vec![
                Message::System {
                    content: "You are a helpful assistant.".to_string(),
                },
                Message::User {
                    content: "Solve this problem: 15 * 24".to_string(),
                },
            ],
        };

        let output =
            ContextState::new(ReasoningEnabled::Yes).chat(&chat, PrefillType::Canonical);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_12_chat_partial_reasoning_prefill() {
        let chat = Chat {
            messages: vec![
                Message::User {
                    content: "Is 97 prime?".to_string(),
                },
            ],
        };

        let output = ContextState::new(ReasoningEnabled::Yes).chat(
            &chat,
            PrefillType::PartialReasoning {
                reasoning_content: "Let me check divisibility...".to_string(),
            },
        );

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_13_chat_full_reasoning_prefill() {
        let chat = Chat {
            messages: vec![
                Message::User {
                    content: "What is 144 / 12?".to_string(),
                },
            ],
        };

        let output = ContextState::new(ReasoningEnabled::Yes).chat(
            &chat,
            PrefillType::FullReasoning {
                reasoning_content: "144 / 12 = 12".to_string(),
                content: "The answer is".to_string(),
            },
        );

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_14_chat_multi_turn_with_assistant_messages() {
        let chat = Chat {
            messages: vec![
                Message::System {
                    content: "You are a math tutor.".to_string(),
                },
                Message::User {
                    content: "What is 5 * 5?".to_string(),
                },
                Message::Assistant {
                    content: "5 * 5 = 25".to_string(),
                    reasoning_content: None,
                },
                Message::User {
                    content: "What about 6 * 6?".to_string(),
                },
            ],
        };

        let output =
            ContextState::new(ReasoningEnabled::No).chat(&chat, PrefillType::Canonical);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_15_chat_multi_turn_with_reasoning() {
        let chat = Chat {
            messages: vec![
                Message::User {
                    content: "Is 13 prime?".to_string(),
                },
                Message::Assistant {
                    content: "Yes, 13 is prime.".to_string(),
                    reasoning_content: Some("Check divisibility: not divisible by 2, 3. Prime.".to_string()),
                },
                Message::User {
                    content: "What about 21?".to_string(),
                },
            ],
        };

        let output =
            ContextState::new(ReasoningEnabled::Yes).chat(&chat, PrefillType::Canonical);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_16_chat_reasoning_disabled_with_reasoning_content() {
        let chat = Chat {
            messages: vec![
                Message::User {
                    content: "Calculate 30 * 12".to_string(),
                },
                Message::Assistant {
                    content: "The answer is 360.".to_string(),
                    reasoning_content: Some("30 * 12 = 30 * (10 + 2) = 300 + 60 = 360".to_string()),
                },
            ],
        };

        let output = ContextState::new(ReasoningEnabled::No).chat(&chat, PrefillType::None);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_17_chat_empty_system_message() {
        let chat = Chat {
            messages: vec![
                Message::System {
                    content: "".to_string(),
                },
                Message::User {
                    content: "Hello!".to_string(),
                },
            ],
        };

        let output =
            ContextState::new(ReasoningEnabled::No).chat(&chat, PrefillType::Canonical);

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_18_chat_complex_multi_turn() {
        let chat = Chat {
            messages: vec![
                Message::System {
                    content: "You are a helpful coding assistant.".to_string(),
                },
                Message::User {
                    content: "How do I reverse a string in Python?".to_string(),
                },
                Message::Assistant {
                    content: "Use slicing: `s[::-1]`".to_string(),
                    reasoning_content: None,
                },
                Message::User {
                    content: "What about in JavaScript?".to_string(),
                },
                Message::Assistant {
                    content: "Use: `str.split('').reverse().join('')`".to_string(),
                    reasoning_content: None,
                },
                Message::User {
                    content: "Thanks! One more: Rust?".to_string(),
                },
            ],
        };

        let output =
            ContextState::new(ReasoningEnabled::No).chat(&chat, PrefillType::Canonical);

        insta::assert_snapshot!(output);
    }
}
