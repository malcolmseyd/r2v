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
    Transform, Rect,
};

use chrono::prelude::*;
use chrono::Duration;

extern crate image;
extern crate nalgebra as na;

// use image::{GenericImageView, Pixel};
use na::DMatrix;

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
    #[arg(long, short, default_value_t = 0.05)]
    parents: f64,
}

impl Args {
    fn num_parents(&self) -> usize {
        ((self.generations) as f64 * self.parents) as usize
    }
}
static ARGS: OnceLock<Args> = OnceLock::new();

// Wrapper function to measure the execution time of a function
fn measure_time<F, R>(func: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let start = Utc::now();
    let result = func();
    let end = Utc::now();
    let duration = end.signed_duration_since(start);
    (duration, result)
}

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
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
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
            Self::Rect { x, y, w, h, color } => {
                // Create tiny skia rect
                Rect {
                    x: *x,
                    y: *y,
                    width: *w,
                    height: *h,
                };

                let path = PathBuilder::from(*x, *y, *w, *h).expect("rect has invalid path");
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

    // Logging info
    // let mut total_time = Duration::zero();
    let mut select_parents_time = Duration::zero();
    let mut crossover_genes_time = Duration::zero();
    let mut mutate_genes_time = Duration::zero();

    for i in 0..ARGS.get().unwrap().generations {
        println!("{i}/{generations}");

        let (duration, parents) = measure_time(|| select_parents(input, prev));
        select_parents_time = select_parents_time + duration;
        // println!("parents seleted");

        let(duration, mut current) = measure_time(|| crossover_genes(&mut rng, parents));
        crossover_genes_time = crossover_genes_time + duration;
        // println!("genes crossed over");

        let duration = measure_time(|| mutate_genes(&mut rng, input, &mut current));
        mutate_genes_time = mutate_genes_time + duration.0;
        // println!("genes mutated");

        prev = current;
    }

    // Print logging info
    // println!("Total time: {} ms", total_time.num_milliseconds());
    println!("Select parents time: {} ms ", select_parents_time.num_milliseconds());
    println!("Crossover genes time: {} ms", crossover_genes_time.num_milliseconds());
    println!("Mutate genes time: {} ms", mutate_genes_time.num_milliseconds());

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

// Returns the Mean Squared Error in the image, with a penalty for the number of shapes
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

    // Penalty term for the number of shapes
    // let shape_penalty = gene.shapes.len() as f64 * 0.5; // Adjust the penalty coefficient as needed
    // let total_error = error + shape_penalty;

    gene.eval.set(Some(error));

    error
}

fn mean(matrix: &DMatrix<f64>) -> f64 {
    matrix.iter().sum::<f64>() / matrix.len() as f64
}

fn variance(matrix: &DMatrix<f64>, mean: f64) -> f64 {
    matrix.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (matrix.len() as f64 - 1.0)
}

fn covariance(matrix1: &DMatrix<f64>, matrix2: &DMatrix<f64>, mean1: f64, mean2: f64) -> f64 {
    matrix1.iter().zip(matrix2.iter())
        .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
        .sum::<f64>() / (matrix1.len() as f64 - 1.0)
}

fn ssim(img1: &DMatrix<f64>, img2: &DMatrix<f64>) -> f64 {
    let k1 = 0.01;
    let k2 = 0.03;
    let l = 255.0;
    let c1 = (k1 * l) * (k1 * l);
    let c2 = (k2 * l) * (k2 * l);

    let mu1 = mean(img1);
    let mu2 = mean(img2);
    let sigma1_sq = variance(img1, mu1);
    let sigma2_sq = variance(img2, mu2);
    let sigma12 = covariance(img1, img2, mu1, mu2);

    ((2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2))
}

fn to_grayscale_matrix(img: &PixmapRef) -> DMatrix<f64> {
    let (width, height) = (img.width() as usize, img.height() as usize);
    let mut data = Vec::with_capacity(width * height);

    for pixel in img.pixels() {
        let gray = 0.299 * pixel.red() as f64 + 0.587 * pixel.green() as f64 + 0.114 * pixel.blue() as f64;
        data.push(gray);
    }

    DMatrix::from_vec(height, width, data)
}

/// Returns the SSIM index for the image, with a penalty for the number of shapes
// fn evaluate(input: PixmapRef, gene: &Gene) -> f64 {
//     if let Some(error) = gene.eval.get() {
//         return error;
//     }

//     let mut output =
//         Pixmap::new(input.width(), input.height()).expect("failed to create output pixmap");
//     gene.render(output.as_mut());

//     let input_matrix = to_grayscale_matrix(&input);
//     let output_matrix = to_grayscale_matrix(&output.as_ref());

//     let ssim_index = ssim(&input_matrix, &output_matrix);

//     // Penalty term for the number of shapes
//     // let shape_penalty = gene.shapes.len() as f64 * 0.1; // Adjust the penalty coefficient as needed

//     let total_error = 1.0 - ssim_index; // SSIM ranges from 0 to 1, so 1 - SSIM will be the error

//     gene.eval.set(Some(total_error));

//     total_error
// }

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

        // Delete existing shape with very low probability
        if rng.gen_ratio(1, 100) {
            if !p.shapes.is_empty() {
                // Remove a random shape, perhaps there is a way to remove shapes that contribute little to the evaluation function
                let index = rng.gen_range(0..p.shapes.len());
                p.shapes.remove(index);
                p.eval.set(None);
            }
        }

        // TODO Dynamic mutations, where mutations become smaller as the generations progress
        // Mutate existing shape with low probability
        if rng.gen_ratio(1, 50) {
            if let Some(shape) = p.shapes.choose_mut(rng) {
                match shape {
                    Shape::Circle { cx, cy, r, color } => {
                        // Adjust position up to +/- 20% of image size
                        *cx += rng.gen_range(-w * 0.2..=w * 0.2);
                        *cy += rng.gen_range(-h * 0.2..=h * 0.2);

                        *cx = cx.clamp(0.0, w);
                        *cy = cy.clamp(0.0, h);

                        // Adjust radius to be +/- 50% current size
                        *r = rng.gen_range(*r * 0.5..=*r * 1.5);
                        *r = r.clamp(0.0, w.max(h) / 10.0);

                        // TODO Probably good to shift the colour slightly as well
                        *color = input.pixel(*cx as u32, *cy as u32).unwrap_or(*color);
                    }
                }
                p.eval.set(None);
            }
        }

        // TODO Dynamic generation of new shapes, where new shapes are generated less frequently as the generations progress
        // Add new shape with some probability
        if rng.gen_ratio(1, 10) {
            // Add circle
            let r = rng.gen_range(0.0..=(w.max(h) / 10.0)); // Adjust max radius for more detailed shapes
            let x = rng.gen_range(-r..w + r);
            let y = rng.gen_range(-r..h + r);

            // Choose pixel at x/y
            if let Some(color) = input.pixel(x as u32, y as u32) {
                // Determine shape to add
                if (rng.gen_ratio(1, 2)) {
                    p.shapes.push(Shape::Square {
                        x,
                        y,
                        w: r,
                        h: r,
                        color,
                    });
                } 
                else 
                {
                    p.shapes.push(Shape::Circle {
                        cx: x,
                        cy: y,
                        r,
                        color,
                    });
                }
                p.eval.set(None);
            }
        }
    }
}