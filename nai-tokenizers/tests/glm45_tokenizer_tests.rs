#[cfg(feature = "glm45_tokenizer")]
mod glm45_tokenizer_tests {
    use nai_tokenizers::glm45_tokenizer::*;

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
