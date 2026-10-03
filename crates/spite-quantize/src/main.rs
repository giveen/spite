//! `spite-quantize` — CLI for converting GGUF models to quantized formats.
//!
//! Usage:
//!   spite-quantize --input base.gguf --output quant.gguf --type Q4_K_M

use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;
use spite_quantize::{QuantizeConfig, QuantType, quantize_model};

#[derive(Parser)]
#[command(name = "spite-quantize", about = "Quantize a GGUF model")]
struct Cli {
    #[arg(short, long)]
    input:   PathBuf,

    #[arg(short, long)]
    output:  PathBuf,

    #[arg(short = 't', long, default_value = "Q4_K_M")]
    r#type:  String,

    #[arg(long, default_value_t = 4)]
    threads: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let quant_type = match cli.r#type.as_str() {
        "Q8_0"   => QuantType::Q8_0,
        "Q4_0"   => QuantType::Q4_0,
        "Q4_K_M" => QuantType::Q4KM,
        "Q4_K_S" => QuantType::Q4KS,
        "Q5_K_M" => QuantType::Q5KM,
        "Q6_K"   => QuantType::Q6K,
        other    => anyhow::bail!("unknown quant type: {other}"),
    };

    let cfg = QuantizeConfig { target: quant_type, n_threads: cli.threads, ..Default::default() };

    println!("quantizing {} → {} ({})", cli.input.display(), cli.output.display(), quant_type.name());
    quantize_model(&cli.input, &cli.output, &cfg)?;
    println!("done");
    Ok(())
}
