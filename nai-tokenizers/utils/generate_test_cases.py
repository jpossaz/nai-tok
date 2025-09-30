"""
Generate test cases for apply_chat_template with various configurations.
Saves input as JSON and rendered output as TXT for Rust comparison.
Uses the zai-org/GLM-4.5 tokenizer from Hugging Face.
"""

import json
import os
from pathlib import Path
from transformers import AutoTokenizer

# Load the GLM-4.5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)

# Create output directories
output_dir = Path("cases")
output_dir.mkdir(exist_ok=True)


def save_test_case(name: str, messages: list, config: dict, output: str):
    """Save test case as JSON input and TXT output"""
    # Combine messages and config into input
    input_data = {
        "messages": messages,
        "config": config,
    }

    # Save JSON input
    json_path = output_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(input_data, f, indent=2)

    # Save TXT output
    txt_path = output_dir / f"{name}.txt"
    with open(txt_path, "w") as f:
        f.write(output)

    print(f"✓ Saved {name}")


def generate_test_cases():
    """Generate all test cases"""

    # Test 1: Basic conversation, no generation prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("01_basic_no_gen_prompt", messages, config, output)

    # Test 2: Basic conversation, with generation prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("02_basic_with_gen_prompt", messages, config, output)

    # Test 3: Thinking enabled
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Solve this math problem: 2 + 2"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("03_thinking_enabled", messages, config, output)

    # Test 4: Thinking disabled
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Solve this math problem: 2 + 2"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("04_thinking_disabled", messages, config, output)

    # Test 5: Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are an expert programmer specializing in Python."},
        {"role": "user", "content": "How do I read a file in Python?"},
        {"role": "assistant", "content": "You can read a file using the `open()` function with a context manager:\n\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"},
        {"role": "user", "content": "What about writing to a file?"},
        {"role": "assistant", "content": "To write to a file, use mode 'w':\n\n```python\nwith open('file.txt', 'w') as f:\n    f.write('Hello, World!')\n```"},
        {"role": "user", "content": "Thanks! Can you show me how to append?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("05_multi_turn", messages, config, output)

    # Test 6: System only
    messages = [
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("06_system_only", messages, config, output)

    # Test 7: No system prompt
    messages = [
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data."},
        {"role": "user", "content": "Can you make a guess?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("07_no_system_prompt", messages, config, output)

    # Test 8: Continue final message (CoT example)
    messages = [
        {"role": "system", "content": "You are a creative writing assistant."},
        {"role": "user", "content": "Write a story."},
        {"role": "assistant", "content": "Beginning:\nOnce upon a time in a distant land"},
        {"role": "user", "content": "Instruction: Make it more mysterious"},
        {"role": "assistant", "content": "The fog rolled in as shadows danced"},
        {"role": "user", "content": "What's your reasoning?"},
        {"role": "assistant", "content": "Understood, here is the chain-of-thought reasoning:\n\n<thinking>\n"},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "continue_final_message": True,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("08_continue_final_message", messages, config, output)

    # Test 9: All combinations - gen_prompt=False, thinking=False
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("09_gen_false_think_false", messages, config, output)

    # Test 10: All combinations - gen_prompt=False, thinking=True
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("10_gen_false_think_true", messages, config, output)

    # Test 11: All combinations - gen_prompt=True, thinking=False
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("11_gen_true_think_false", messages, config, output)

    # Test 12: All combinations - gen_prompt=True, thinking=True
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("12_gen_true_think_true", messages, config, output)

    # Test 13: Empty assistant message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": ""},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("13_empty_assistant", messages, config, output)

    # Test 14: Long conversation
    messages = [
        {"role": "system", "content": "You are a knowledgeable history teacher."},
        {"role": "user", "content": "Who was Julius Caesar?"},
        {"role": "assistant", "content": "Julius Caesar was a Roman general and statesman who played a critical role in the transformation of the Roman Republic into the Roman Empire."},
        {"role": "user", "content": "When did he live?"},
        {"role": "assistant", "content": "He lived from 100 BC to 44 BC."},
        {"role": "user", "content": "How did he die?"},
        {"role": "assistant", "content": "He was assassinated by a group of senators on the Ides of March (March 15) in 44 BC."},
        {"role": "user", "content": "Who were the main conspirators?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("14_long_conversation", messages, config, output)

    # Test 15: User message only
    messages = [
        {"role": "user", "content": "Hello!"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("15_user_only", messages, config, output)

    # Test 16: Complex CoT scenario (from original example)
    system = "You are a creative writer who thinks carefully about narrative structure."
    begin = "The old lighthouse stood alone on the cliff"
    inst = "Add more atmospheric details"
    cont = "Waves crashed against the rocks below, sending spray high into the air. The wind howled through the broken windows."
    cot_prompt = "Explain your creative reasoning process step by step."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Write."},
        {"role": "assistant", "content": "Beginning:\n" + begin},
        {"role": "user", "content": "Instruction: " + inst},
        {"role": "assistant", "content": cont},
        {"role": "user", "content": cot_prompt},
        {"role": "assistant", "content": "Understood, here is the chain-of-thought reasoning that would lead to generating the continuation in my previous message given my first message and your previous instruction:\n\n<thinking>\n"},
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "continue_final_message": True,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("16_complex_cot", messages, config, output)

    # Test 17: Assistant with reasoning_content (thinking with content)
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 15 * 24?"},
        {
            "role": "assistant",
            "reasoning_content": "Let me break this down:\n- 15 * 24 = 15 * (20 + 4)\n- = (15 * 20) + (15 * 4)\n- = 300 + 60\n- = 360",
            "content": "The answer is 360."
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("17_reasoning_content_think_disabled", messages, config, output)

    # Test 18: Assistant with reasoning_content and enable_thinking=True
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 15 * 24?"},
        {
            "role": "assistant",
            "reasoning_content": "Let me break this down:\n- 15 * 24 = 15 * (20 + 4)\n- = (15 * 20) + (15 * 4)\n- = 300 + 60\n- = 360",
            "content": "The answer is 360."
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("18_reasoning_content_think_enabled", messages, config, output)

    # Test 19: Multi-turn with reasoning_content
    messages = [
        {"role": "system", "content": "You are a logical reasoning assistant."},
        {"role": "user", "content": "Is 17 prime?"},
        {
            "role": "assistant",
            "reasoning_content": "To check if 17 is prime:\n- Check divisibility by 2: 17/2 = 8.5 (not divisible)\n- Check divisibility by 3: 17/3 = 5.67 (not divisible)\n- Check divisibility by 5: 17/5 = 3.4 (not divisible)\n- We only need to check up to sqrt(17) ≈ 4.12\n- No divisors found, so 17 is prime",
            "content": "Yes, 17 is a prime number."
        },
        {"role": "user", "content": "What about 18?"},
        {
            "role": "assistant",
            "reasoning_content": "To check if 18 is prime:\n- Check divisibility by 2: 18/2 = 9 (divisible!)\n- Since 18 = 2 × 9, it has divisors other than 1 and itself",
            "content": "No, 18 is not prime. It's divisible by 2, 3, 6, and 9."
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("19_multi_turn_reasoning", messages, config, output)

    # Test 20: Reasoning_content with generation prompt
    messages = [
        {"role": "user", "content": "Calculate 7 squared."},
        {
            "role": "assistant",
            "reasoning_content": "7 squared means 7 × 7 = 49",
            "content": "7² = 49"
        },
        {"role": "user", "content": "Now do 8 squared."},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("20_reasoning_with_gen_prompt", messages, config, output)

    # Test 21: Only reasoning_content, no content field
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Think through the solution to 2+2."},
        {
            "role": "assistant",
            "reasoning_content": "2 + 2 = 4. This is basic addition.",
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("21_reasoning_only_no_content", messages, config, output)

    # Test 22: Empty reasoning_content
    messages = [
        {"role": "user", "content": "Hello!"},
        {
            "role": "assistant",
            "reasoning_content": "",
            "content": "Hi there!"
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("22_empty_reasoning_content", messages, config, output)


if __name__ == "__main__":
    print("Generating test cases for GLM-4.5 tokenizer...\n")
    generate_test_cases()
    print(f"\n✅ All test cases generated in {output_dir}/")
    print("Each test case has:")
    print("  - .json file with messages and config")
    print("  - .txt file with rendered output")