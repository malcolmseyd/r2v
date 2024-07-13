use rand::prelude::*;
use rayon::prelude::*;

use std::{
    cell::Cell,
    fs::File,
    io::{stdin, stdout, Cursor, Read, Write},
    sync::OnceLock,
};

use anyhow::{Context, Ok, Result};
use clap::Parser;
use image::io::Reader as ImageReader;
use tiny_skia::{
    Color, IntSize, Paint, PathBuilder, Pixmap, PixmapMut, PixmapRef, PremultipliedColorU8, Shader,
    Transform,
};

#[derive(Parser, Debug, Default)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filename of the SVG graphic to output
    #[arg(long, short)]
    outfile: String,

    /// Filename of the raster graphic to convert
    infile: Option<String>,

    /// The number of generations to simulate
    #[arg(long, short, default_value_t = 1000)]
    generations: usize,

    /// The size of each generation
    #[arg(long, short, default_value_t = 1000)]
    size: usize,

    // A number in range [0,1] representing the percentage of each generation to keep as parents
    #[arg(long, short, default_value_t = 0.5)]
    parents: f64,
}

impl Args {
    fn num_parents(&self) -> usize {
        ((self.generations) as f64 * self.parents) as usize
    }
}
static ARGS: OnceLock<Args> = OnceLock::new();

fn main() -> Result<()> {
    ARGS.set(Args::parse()).unwrap();

    let input: &mut dyn Read = match &ARGS.get().unwrap().infile {
        Some(path) => &mut File::open(path).context("failed to open input file")?,
        None => &mut stdin(),
    };

    let output: &mut dyn Write = if ARGS.get().unwrap().outfile == "-" {
        &mut stdout()
    } else {
        &mut File::create(ARGS.get().unwrap().outfile.clone())
            .context("failed to open output file")?
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

#[derive(Clone)]
enum Shape {
    /// Circle centered on `cx` and `cy` with radius `r`
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        color: PremultipliedColorU8,
    },
}

impl Shape {
    fn render(&self, output: &mut PixmapMut) {
        match self {
            Self::Circle { cx, cy, r, color } => {
                let path = PathBuilder::from_circle(*cx, *cy, *r).expect("circle has invalid path");
                // HACK: ignore the alpha for now
                let paint = Paint {
                    shader: Shader::SolidColor(Color::from_rgba8(
                        color.red(),
                        color.green(),
                        color.blue(),
                        u8::MAX,
                    )),
                    ..Paint::default()
                };
                output.fill_path(
                    &path,
                    &paint,
                    tiny_skia::FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
        }
    }

    fn to_svg(&self, w: &mut impl Write) -> Result<()> {
        match self {
            Self::Circle { cx, cy, r, color } => {
                write!(
                    w,
                    r#"<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgb({} {} {})"/>"#,
                    color.red(),
                    color.green(),
                    color.blue(),
                )
            }
        }?;
        Ok(())
    }
}

#[derive(Clone)]
struct Gene {
    shapes: Vec<Shape>,
    eval: Cell<Option<f64>>,
}

impl Gene {
    fn render(&self, mut output: PixmapMut) {
        for shape in self.shapes.iter() {
            shape.render(&mut output);
        }
    }

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
    let mut rng = StdRng::from_entropy();

    let mut prev: Population = init_generation();

    let generations = ARGS.get().unwrap().generations;

    for i in 0..ARGS.get().unwrap().generations {
        println!("{i}/{generations}");

        let parents = select_parents(input, prev);
        println!("parents seleted");

        let mut current = crossover_genes(&mut rng, parents);
        println!("genes crossed over");

        mutate_genes(&mut rng, input, &mut current);
        println!("genes mutated");

        prev = current;
    }

    let best = &prev[0];

    let mut output = Vec::new();
    best.to_svg(input, &mut output)?;

    Ok(output)
}

fn init_generation() -> Population {
    let mut init: Population = Vec::new();

    for _ in 0..ARGS.get().unwrap().size {
        let shapes = Vec::new();
        init.push(Gene {
            shapes,
            eval: Cell::new(None),
        });
    }

    init
}

fn select_parents(input: PixmapRef, prev: Population) -> Population {
    let mut results = prev
        // .into_iter()
        .into_par_iter()
        .map(|gene| (evaluate(input, &gene), gene))
        .collect::<Vec<_>>();

    // lowest error first
    results.sort_by(|(score_a, _), (score_b, _)| score_a.total_cmp(score_b));
    results.truncate(ARGS.get().unwrap().num_parents());

    println!("{} - {}", results[0].0, results[results.len() - 1].0);

    results.into_iter().map(|(_, gene)| gene).collect()
}

/// Returns the Mean Squared Error in the image
fn evaluate(input: PixmapRef, gene: &Gene) -> f64 {
    if let Some(error) = gene.eval.get() {
        return error;
    }

    let mut output =
        Pixmap::new(input.width(), input.height()).expect("failed to create output pixmap");
    gene.render(output.as_mut());

    let mut error = 0.0f64;
    for (i, o) in input.pixels().iter().zip(output.pixels().iter()) {
        error += (i.red() as i32 - o.red() as i32).pow(2) as f64;
        error += (i.green() as i32 - o.green() as i32).pow(2) as f64;
        error += (i.blue() as i32 - o.blue() as i32).pow(2) as f64;
    }
    error /= (output.height() * output.width()) as f64;

    gene.eval.set(Some(error));

    error
}

fn crossover_genes(rng: &mut impl Rng, parents: Population) -> Population {
    let mut current = parents;

    for _ in current.len()..ARGS.get().unwrap().size {
        // TODO: share genes from parents randomly
        current.push(
            current
                .choose(rng)
                .expect("parents must not be empty")
                .clone(),
        );
    }

    current
}

fn mutate_genes(rng: &mut impl Rng, input: PixmapRef, current: &mut Population) {
    let w = input.width() as f32;
    let h = input.height() as f32;

    // Don't forget to reset p.eval if you change anything!
    for p in current.iter_mut() {
        // add shape
        if rng.gen_ratio(1, 10) {
            // add circle
            let &color = input.pixels().choose(rng).expect("input is empty");
            if color.alpha() == 0 {
                continue;
            }

            let r = rng.gen_range(0.0..=(w.max(h)));
            p.shapes.push(Shape::Circle {
                cx: rng.gen_range(-r..w + r),
                cy: rng.gen_range(-r..h + r),
                r,
                color,
            });
            p.eval.set(None)
        }

        // TODO: mutate existing shape with low probability
        // TODO: delete existing shape with very low probability
    }
}
