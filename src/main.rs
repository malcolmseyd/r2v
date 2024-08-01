use cached::proc_macro::cached;
use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;

use std::{
    cell::Cell,
    cmp::max_by_key,
    fs::File,
    hash::Hash,
    io::{stdin, stdout, BufWriter, Cursor, Read, Write},
    sync::{
        atomic::{self, AtomicBool},
        OnceLock,
    },
    thread,
    time::Duration,
};

use anyhow::{anyhow, Context, Ok, Result};
use clap::Parser;
use image::{io::Reader as ImageReader, ImageBuffer, Rgba};
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

    /// The maximum time to run the simulation (in seconds)
    #[arg(long, short = 't')]
    max_time: Option<f64>,

    /// The size of each generation
    #[arg(long, short, default_value_t = 1000)]
    size: usize,

    ///  A number in range [0,1] representing the percentage of each generation to keep as parents
    #[arg(long, short, default_value_t = 0.05)]
    parents: f64,

    /// The maximum dimension to downscale the image to, clamping either the height or width
    #[arg(long, short, default_value_t = 64)]
    max_dimension: u32,
}

impl Args {
    fn num_parents(&self) -> usize {
        ((self.generations) as f64 * self.parents) as usize
    }
}
static ARGS: OnceLock<Args> = OnceLock::new();

static STOP: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
    ARGS.set(Args::parse()).unwrap();

    if let Some(max_seconds) = ARGS.get().unwrap().max_time {
        thread::spawn(move || {
            thread::sleep(Duration::from_millis((max_seconds * 1000.0) as u64));
            STOP.store(true, atomic::Ordering::Relaxed);
        });
    }

    let input: &mut dyn Read = match &ARGS.get().unwrap().infile {
        Some(path) => &mut File::open(path)
            .with_context(|| anyhow!("failed to open input file: {:?}", path))?,
        None => &mut stdin(),
    };

    let output: &mut dyn Write = if ARGS.get().unwrap().outfile == "-" {
        &mut stdout()
    } else {
        let path = &ARGS.get().unwrap().outfile;
        &mut File::create(path)
            .with_context(|| anyhow!("failed to open output file: {:?}", path))?
    };

    let mut input_buf = Vec::new();
    input.read_to_end(&mut input_buf)?;

    // source of truth for original image
    let img = ImageReader::new(Cursor::new(input_buf))
        .with_guessed_format()?
        .decode()?
        .into_rgba8();

    // downscaled, ready for use in the algorithm
    let pixmap = to_pixmap(&img).context("failed to translate input to pixmap")?;

    let best_gene = run(pixmap.as_ref())?;
    best_gene
        .to_svg(&mut BufWriter::new(output), img.width(), img.height())
        .context("failed to serialize SVG to file")?;

    Ok(())
}

fn to_pixmap(img: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<Pixmap> {
    let max_dimension = ARGS.get().unwrap().max_dimension;

    let img = if max_dimension >= std::cmp::max(img.height(), img.width()) {
        img.to_owned()
    } else {
        let (nwidth, nheight) = if img.width() > img.height() {
            let nwidth: u32 = max_dimension;
            let nheight = ((nwidth as f64) / (img.width() as f64) * img.height() as f64) as u32;
            (nwidth, nheight)
        } else {
            let nheight: u32 = max_dimension;
            let nwidth = ((nheight as f64) / (img.height() as f64) * img.width() as f64) as u32;
            (nwidth, nheight)
        };

        image::imageops::resize(
            img,
            nwidth,
            nheight,
            image::imageops::FilterType::CatmullRom,
        )
    };

    let pixmap = {
        let size = IntSize::from_wh(img.width(), img.height())
            .context("failed to construct input pixbuf size")?;
        Pixmap::from_vec(img.into_vec(), size).context("failed to construct input pixbuf")?
    };

    Ok(pixmap)
}

#[derive(Clone, PartialEq)]
enum Shape {
    /// Circle centered on `cx` and `cy` with radius `r`
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        color: PremultipliedColorU8,
    },
}

// marker trait
impl Eq for Shape {}

impl Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Circle { cx, cy, r, color } => {
                cx.to_bits().hash(state);
                cy.to_bits().hash(state);
                r.to_bits().hash(state);
                color.red().hash(state);
                color.green().hash(state);
                color.blue().hash(state);
                color.alpha().hash(state);
            }
        }
    }
}

impl Shape {
    fn render(&self, output: &mut PixmapMut) {
        match self {
            Self::Circle { cx, cy, r, color } => {
                let path = PathBuilder::from_circle(*cx, *cy, *r).expect("circle has invalid path");
                let paint = Paint {
                    shader: Shader::SolidColor(Color::from_rgba8(
                        color.red(),
                        color.green(),
                        color.blue(),
                        color.alpha(),
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

    fn to_svg(&self, w: &mut impl Write, scale: f32) -> Result<()> {
        match self {
            Self::Circle { cx, cy, r, color } => {
                write!(
                    w,
                    r#"<circle cx="{}" cy="{}" r="{}" fill="rgb({} {} {})"/>"#,
                    cx * scale,
                    cy * scale,
                    r * scale,
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

impl PartialEq for Gene {
    fn eq(&self, other: &Self) -> bool {
        self.shapes == other.shapes
    }
}

impl Eq for Gene {}

impl Hash for Gene {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shapes.hash(state);
    }
}

#[cached(size = 10000)]
fn render_shapes(width: u32, height: u32, shapes: Vec<Shape>) -> Pixmap {
    if shapes.is_empty() {
        return Pixmap::new(width, height).unwrap();
    }
    let mut p = render_shapes(width, height, shapes[..shapes.len() - 1].to_vec());
    shapes[shapes.len() - 1].render(&mut p.as_mut());
    p
}

impl Gene {
    fn new(shapes: Vec<Shape>) -> Self {
        Gene {
            shapes,
            eval: Cell::new(None),
        }
    }

    fn render(&self, width: u32, height: u32) -> Pixmap {
        render_shapes(width, height, self.shapes.clone())
    }

    fn to_svg(&self, w: &mut impl Write, width: u32, height: u32) -> Result<()> {
        write!(
            w,
            r#"<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
            width, height, width, height
        )?;

        let image_max_dimension = std::cmp::max(width, height);
        let max_dimension = ARGS.get().unwrap().max_dimension;
        // if <1 then we didn't scale it, clamp to 0
        let scale = ((image_max_dimension as f64) / (max_dimension as f64)).max(1.0);

        for shape in self.shapes.iter() {
            shape.to_svg(w, scale as f32)?
        }
        write!(w, r#"</svg>"#)?;
        Ok(())
    }
}

type Population = Vec<Gene>;

fn run(input: PixmapRef) -> Result<Gene> {
    let mut rng = SmallRng::from_entropy();
    let mut prev: Population = init_generation();

    let generations = ARGS.get().unwrap().generations;
    for i in 0..ARGS.get().unwrap().generations {
        if STOP.load(atomic::Ordering::Relaxed) {
            println!("Hit time limit, stopping early...");
            break;
        }
        println!("{i}/{generations}");
        let parents = select_parents(input, prev);
        let mut current = crossover_genes(&mut rng, parents);
        mutate_genes(&mut rng, input, &mut current);
        prev = current;
    }

    Ok(prev[0].clone())
}

fn init_generation() -> Population {
    let mut init: Population = Vec::new();

    for _ in 0..ARGS.get().unwrap().size {
        let shapes = Vec::new();
        init.push(Gene::new(shapes));
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

    println!("eval {} - {}", results[0].0, results[results.len() - 1].0);

    results.into_iter().map(|(_, gene)| gene).collect()
}

/// Returns the Mean Squared Error in the image
fn evaluate(input: PixmapRef, gene: &Gene) -> f64 {
    if let Some(error) = gene.eval.get() {
        return error;
    }

    let output = gene.render(input.width(), input.height());

    let mut error = 0.0f64;
    for (i, o) in input.pixels().iter().zip(output.pixels().iter()) {
        error += (i.red() as i32 - o.red() as i32).pow(2) as f64;
        error += (i.green() as i32 - o.green() as i32).pow(2) as f64;
        error += (i.blue() as i32 - o.blue() as i32).pow(2) as f64;
    }

    error /= output.pixels().len() as f64;

    gene.eval.set(Some(error));

    error
}

fn crossover_genes(rng: &mut impl Rng, parents: Population) -> Population {
    let mut current = parents;

    for _ in current.len()..ARGS.get().unwrap().size {
        // elitism, just copy the parent unmodified
        if rng.gen_ratio(4, 5) {
            current.push(
                current
                    .choose(rng)
                    .expect("parents must not be empty")
                    .clone(),
            );
            continue;
        }

        let mut shapes = Vec::new();

        let (a, b) = current
            .choose_multiple(rng, 2)
            .next_tuple()
            .expect("crossover needs at least 2 genes");

        for (sa, sb) in a.shapes.iter().zip(b.shapes.iter()) {
            if rng.gen_ratio(1, 2) {
                shapes.push(sa.clone())
            } else {
                shapes.push(sb.clone())
            }
        }

        if a.shapes.len() != b.shapes.len() && rng.gen_ratio(1, 2) {
            let biggest = max_by_key(a, b, |x| x.shapes.len());
            shapes.extend(biggest.shapes.clone().into_iter().skip(shapes.len()));
        }

        current.push(Gene::new(shapes));
    }

    current
}

fn mutate_genes(rng: &mut impl Rng, input: PixmapRef, current: &mut Population) {
    let w = input.width() as f32;
    let h = input.height() as f32;

    // Don't forget to reset p.eval if you change anything!
    for p in current.iter_mut() {
        // Delete existing shape with very low probability
        if rng.gen_ratio(1, 100) && !p.shapes.is_empty() {
            // Remove a random shape, perhaps there is a way to remove shapes that contribute little to the evaluation function
            let index = rng.gen_range(0..p.shapes.len());
            p.shapes.remove(index);
            p.eval.set(None);
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

            // randomly sample color inside the shape's bounding box
            let inside_x: u32 =
                rng.gen_range((x - r).clamp(0.0, w) as u32..=(x + r).clamp(0.0, w) as u32);
            let inside_y: u32 =
                rng.gen_range((y - r).clamp(0.0, h) as u32..=(y + r).clamp(0.0, h) as u32);
            if let Some(color) = input.pixel(inside_x, inside_y) {
                p.shapes.push(Shape::Circle {
                    cx: x,
                    cy: y,
                    r,
                    color,
                });
                p.eval.set(None);
            }
        }
    }
}
