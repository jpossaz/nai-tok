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
}
