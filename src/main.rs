use std::{
    fs::File,
    io::{stdin, stdout, Cursor, Read, Write},
};

use anyhow::{Context, Ok, Result};
use clap::Parser;
use image::{io::Reader as ImageReader, RgbaImage};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filename of the SVG graphic to output
    #[arg(long, short)]
    outfile: String,

    /// Filename of the raster graphic to convert
    infile: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input: &mut dyn Read = match &args.infile {
        Some(path) => &mut File::open(path).context("failed to open input file")?,
        None => &mut stdin(),
    };

    let output: &mut dyn Write = if args.outfile == "-" {
        &mut stdout()
    } else {
        &mut File::create(args.outfile).context("failed to open output file")?
    };

    let mut input_buf = Vec::new();
    input.read_to_end(&mut input_buf)?;

    let img = ImageReader::new(Cursor::new(input_buf))
        .with_guessed_format()?
        .decode()?
        .into_rgba8();

    let output_buf = run(img)?;
    output.write_all(&output_buf)?;

    Ok(())
}

const GENERATIONS: u64 = 1000;

enum Shape {
    /// Circle centered on cx and cy
    Circle { cx: f32, cy: f32, radius: f32 },
}

struct Gene {
    shapes: Vec<Shape>,
}

type Population = Vec<Gene>;

fn run(input: RgbaImage) -> Result<Vec<u8>> {
    // TODO: use tiny_skia to draw circles of random color and location onto a Pixmap
    // TODO: use MSE to calculate total error
    let mut prev: Population = init_generation();

    for _ in 0..GENERATIONS {
        let parents = select_parents(prev);
        let mut current = crossover_genes(parents);
        mutate_genes(&mut current);
        prev = current;
    }

    let output = Vec::new();
    Ok(output)
}

fn init_generation() -> Population {
    todo!()
}

fn select_parents(prev: Population) -> Population {
    // TODO: evaluate fitness of parents and prune the worst
    prev
}

fn crossover_genes(parents: Population) -> Population {
    // TODO: share objects from parents randomly
    parents
}

fn mutate_genes(current: &mut Population) {
    // TODO: mutate the new generation randomly
}
