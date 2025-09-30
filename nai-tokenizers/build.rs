use std::fs;
use std::path::Path;

fn main() {
    let tokenizers_dir = "tokenizers";

    // Create tokenizers directory if it doesn't exist
    fs::create_dir_all(tokenizers_dir).expect("Failed to create tokenizers directory");

    // Download GLM-4.5 tokenizer files
    download_file(
        "https://huggingface.co/zai-org/GLM-4.5/resolve/main/tokenizer.json",
        &format!("{}/glm-4.5-tokenizer.json", tokenizers_dir)
    );

    download_file(
        "https://huggingface.co/zai-org/GLM-4.5/resolve/main/tokenizer_config.json",
        &format!("{}/glm-4.5-tokenizer-config.json", tokenizers_dir)
    );

    println!("cargo:rerun-if-changed=build.rs");
}

fn download_file(url: &str, destination: &str) {
    let path = Path::new(destination);

    // Skip download if file already exists
    if path.exists() {
        println!("cargo:warning=File already exists: {}", destination);
        return;
    }

    println!("cargo:warning=Downloading {} to {}", url, destination);

    let response = ureq::get(url)
        .call()
        .expect(&format!("Failed to download {}", url));

    let mut file = fs::File::create(path)
        .expect(&format!("Failed to create file {}", destination));

    std::io::copy(&mut response.into_reader(), &mut file)
        .expect(&format!("Failed to write to {}", destination));

    println!("cargo:warning=Successfully downloaded {}", destination);
}