### The following outlines how to compile and run our genetic algorithm on just about any input image you could think of.

# Compilation

1. Please ensure that you have rust and cargo installed on the compiling computer. They can be easily installed by following the official steps provided here https://www.rust-lang.org/tools/install

2. Clone this repo, and navigate to the within the `r2v` directory.

3. Run the command `cargo run --release` to build the program with maximal performance. It can of course be built in any mode, but release mode will provide decreased runtimes for testing.

4. Done! Now onto running the built program.

# Running

1. Navigate to `r2v/target/release`.

2. Run the command `./r2v --outfile output.svg ../../datasets/{INPUT_IMAGE}` to run the algorithm with default parameters.

3. In order to tune the generation, append the following flags:

| Parameter            | Value                       | Description                              |
|----------------------|-----------------------------|------------------------------------------|
| -g                   | {GENERATION_COUNT}          | Number of generations                    |
| -s                   | {GENERATION_SIZE}           | Size of each generation                  |
| -p                   | {ELITISM_RATIO}             | Ratio of elites kept per generation      |
| -m                   | {DOWNSCALING_MAX_DIMENSION} | Maximum dimension for preprocessing downscaling step       |
| -t                   | {MAX_PROGRAM_TIME}          | Maximum time allowed for the program in seconds     |


4. Here is an example with each flag set: `./r2v --outfile output.svg ../../datasets/{INPUT_IMAGE} -g 1500 -s 1200 -p 0.1 -m 96 -t 240`

5. Have fun! For quick tests we recommend leaving parameters close to default and setting a time limit as necessary, around 30-60 seconds usually produces good results on most computers. 
