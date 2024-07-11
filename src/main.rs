use rand::prelude::*;
use rayon::prelude::*;

use std::{
    fs::File,
    io::{stdin, stdout, Cursor, Read, Write},
};

use anyhow::{Context, Ok, Result};
use clap::Parser;
use image::{io::Reader as ImageReader, Rgba};
use tiny_skia::{IntSize, Pixmap, PixmapRef};

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

    let pixmap = {
        let size = IntSize::from_wh(img.width(), img.height())
            .context("failed to construct input pixbuf size")?;
        Pixmap::from_vec(img.into_vec(), size).context("failed to construct input pixbuf")?
    };

    let output_buf = run(pixmap.as_ref())?;
    output.write_all(&output_buf)?;

    Ok(())
}

const GENERATIONS: usize = 1000;
const GENERATION_SIZE: usize = 1000;
const GENERATION_PARENTS: usize = GENERATION_SIZE * 5 / 10;

enum Shape {
    /// Circle centered on `cx` and `cy` with radius `r`
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        color: Rgba<u8>,
    },
}

impl Shape {
    fn to_svg(&self, w: &mut impl Write) -> Result<()> {
        match self {
            Self::Circle { cx, cy, r, color } => {
                write!(
                    w,
                    r#"<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgb({} {} {})"/>"#,
                    color.0[0], color.0[1], color.0[2]
                )
            }
        }?;
        Ok(())
    }
}

struct Gene {
    shapes: Vec<Shape>,
}

impl Gene {
    fn to_svg(&self, input: PixmapRef, w: &mut impl Write) -> Result<()> {
        write!(
            w,
            r#"<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
            input.width(),
            input.height(),
            input.width(),
            input.height()
        )?;
        for shape in self.shapes.iter() {
            shape.to_svg(w)?
        }
        write!(w, r#"</svg>"#)?;
        Ok(())
    }
}

type Population = Vec<Gene>;

fn run(input: PixmapRef) -> Result<Vec<u8>> {
    // TODO: use tiny_skia to draw circles of random color and location onto a Pixmap
    // TODO: use MSE to calculate total error
    let mut prev: Population = init_generation();
    let mut rng = StdRng::from_entropy();

    for _ in 0..GENERATIONS {
        let parents = select_parents(input, prev);
        let mut current = crossover_genes(&mut rng, parents);
        mutate_genes(&mut rng, input, &mut current);
        prev = current;
    }

    // TODO: properly select the best for output
    let best = &prev[0];

    let mut output = Vec::new();
    best.to_svg(input, &mut output)?;

    Ok(output)
}

fn init_generation() -> Population {
    let mut init: Population = Vec::new();

    for _ in 0..GENERATION_SIZE {
        let shapes = Vec::new();
        init.push(Gene { shapes });
    }

    init
}

fn select_parents(input: PixmapRef, prev: Population) -> Population {
    let mut results = prev
        .into_par_iter()
        .map(|gene| (evaluate(input, &gene), gene))
        .collect::<Vec<_>>();

    results.sort_by(|(score_a, _), (score_b, _)| score_a.total_cmp(score_b));

    // TODO: prune the worst

    results.into_iter().map(|(_, gene)| gene).collect()
}

fn evaluate(input: PixmapRef, gene: &Gene) -> f32 {
    // TODO: calculate error on pixels
    0.0
}

fn crossover_genes(rng: &mut impl Rng, parents: Population) -> Population {
    // TODO: share genes from parents randomly
    parents
}

fn mutate_genes(rng: &mut impl Rng, input: PixmapRef, current: &mut Population) {
    let w = input.width() as f32;
    let h = input.height() as f32;
    for p in current.iter_mut() {
        // add shape
        if rng.gen_ratio(1, 10) {
            // add circle
            let color: Rgba<u8> = rng.gen::<[u8; 4]>().into();

            let r = rng.gen_range(0.0..=(w.max(h)));
            p.shapes.push(Shape::Circle {
                cx: rng.gen_range(-r..w + r),
                cy: rng.gen_range(-r..h + r),
                r,
                color,
            })
        }
    }
}
