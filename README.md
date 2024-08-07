
# Raster 2 Vector

r2v is a raster-to-vector image converter, powered by a genetic algorithm!

## Building

1. Please ensure that you have rust and cargo installed on the compiling computer. They can be easily installed by following the official steps provided here https://www.rust-lang.org/tools/install

2. Clone this repo, and navigate to the within the `r2v` directory.

3. Run the following commands:

```console
$ cargo build --release
    Finished `release` profile [optimized + debuginfo] target(s) in 0.08s
$ ls -la target/release/r2v
-rwxr-xr-x 2 malcolm wheel 105122048 Jul 31 13:11 target/release/r2v
$ target/release/r2v --version
r2v 0.1.0
```

The binary lives in `target/release/r2v`, so either reference it directly or move it to your PATH.

## Running

This assumes that the binary is in your path. For INFILE, we have samples in
the `dataset` folder to use for testing.

```console
$ r2v
Usage: r2v [OPTIONS] --outfile <OUTFILE> [INFILE]

Arguments:
  [INFILE]  Filename of the raster graphic to convert

Options:
  -o, --outfile <OUTFILE>              Filename of the SVG graphic to output
  -g, --generations <GENERATIONS>      The number of generations to simulate [default: 1000]
  -t, --max-time <MAX_TIME>            The maximum time to run the simulation (in seconds)
  -s, --size <SIZE>                    The size of each generation [default: 1000]
  -p, --parents <PARENTS>              A number in range [0,1] representing the percentage of each generation to keep as parents [default: 0.05]
  -m, --max-dimension <MAX_DIMENSION>  The maximum dimension to downscale the image to, clamping either the height or width [default: 64]
  -h, --help                           Print help
  -V, --version                        Print version
```

For example:

```console
$ r2v -p 0.05 -g 1000 -s 1000 -m 150 -o martlet.svg datasets/Martlet-blue.jpg
eval 113262.55912643678 - 113262.55912643678
Elapsed time: 0.010649 seconds, 0/1000
eval 107111.6283678161 - 113095.02174712643
Elapsed time: 0.016625 seconds, 1/1000
... snip ...
Elapsed time: 95.738017 seconds, 997/1000
eval 1029.1683678160919 - 1033.1570114942529
Elapsed time: 95.962300 seconds, 998/1000
eval 1026.4571494252873 - 1030.9492873563217
Elapsed time: 96.082652 seconds, 999/1000
Done! Final score: 1026.4571494252873
$ open martlet.svg
```
4. Here is an example with each flag set: `./r2v --outfile output.svg ../../datasets/{INPUT_IMAGE} -g 1500 -s 1200 -p 0.1 -m 96 -t 240`

5. Have fun! For quick tests we recommend leaving parameters close to default and setting a time limit as necessary, around 30-60 seconds usually produces good results on most computers. 
