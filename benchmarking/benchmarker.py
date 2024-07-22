import subprocess
import time
import matplotlib.pyplot as plot
# import numpy as np
from matplotlib.ticker import MaxNLocator

# Paths
executable = "../target/release/r2v"
discord_small = "../datasets/discord_small.jpg"

# General variables
current_time = time.time()
current_file_name = "discord_small.jpg"
current_file_path = discord_small
current_parameter = "g"

default_generations = 1000
default_size = 1000
default_time = 30
default_elitism = 0.05

run_results = []
score_samples = []
benchmark_runs = []
sample_times = [0, 5, 10, 15, 20, 25]

# Benchmark variables
generations_range = [250, 500, 750, 1000, 1250]
# size_range = [500, 1000, 1500, 2000, 2500]
size_range = [500, 1000]
elitism_range = [0.01, 0.05, 0.1, 0.15, 0.2]

# Get parameters to run as input
do_generations = None
do_size = None
do_elitism = None

while True:
    do_generations = input("Run benchmark on generations? (Y/n): ").strip().lower()
    if do_generations in ('y', ''):
        do_generations = True
        break
    elif do_generations == 'n':
        do_generations = False
        break

while True:
    do_size = input("Run benchmark on size? (Y/n): ").strip().lower()
    if do_size in ('y', ''):
        do_size = True
        break
    elif do_size == 'n':
        do_size = False
        break

while True:
    do_elitism = input("Run benchmark on elitism? (Y/n): ").strip().lower()
    if do_elitism in ('y', ''):
        do_elitism = True
        break
    elif do_elitism == 'n':
        do_elitism = False
        break

# Run benchmark for each specified parameter
if do_generations:
    current_parameter = "g"

    for index, generations in enumerate(generations_range):
        current_time = time.time()

        # Run the r2v program with the specified image, adjusting the selected parameter over the specified range
        benchmark = subprocess.Popen([executable, "--outfile", "gen_output" + str(index) + ".svg", current_file_path, "-g", str(generations), "-s", str(default_size), "-p", str(default_elitism), "-t", str(default_time)], stdout=subprocess.PIPE)

        # Await the completion of the benchmark
        print("Benchmark started on parameter " + current_parameter + " = " + str(generations) + " (" + current_file_name + "): 0.00 seconds elapsed")
        while benchmark.poll() is None:
            # Stream the output of the benchmark
            # score_samples.append(benchmark.stdout.readlines()[-1].decode("utf-8").split(" ")[-1].strip())
            # print(benchmark.stdout.readlines()[-1])
            time.sleep(5)
            print("Benchmark running on parameter " + current_parameter + " = " + str(generations) + " (" + current_file_name + "): {:.2f} seconds elapsed".format(time.time() - current_time))

        # Extract the results of the benchmark

if do_size:
    current_parameter = "s"

    for index, size in enumerate(size_range):
        current_time = time.time()

        # Run the r2v program with the specified image, adjusting the selected parameter over the specified range
        benchmark = subprocess.Popen([executable, "--outfile", "gen_output" + str(index) + ".svg", current_file_path, "-g", str(default_generations), "-s", str(size), "-p", str(default_elitism), "-t", str(default_time)], stdout=subprocess.PIPE)

        # Await the completion of the benchmark
        print("Benchmark started on parameter " + current_parameter + " = " + str(size) + " (" + current_file_name + "): 0.00 seconds elapsed")
        while benchmark.poll() is None:
            # Stream the output of the benchmark
            # score_samples.append(benchmark.stdout.readlines()[-1].decode("utf-8").split(" ")[-1].strip())
            # print(benchmark.stdout.readlines()[-1])
            time.sleep(5)
            print("Benchmark running on parameter " + current_parameter + " = " + str(size) + " (" + current_file_name + "): {:.2f} seconds elapsed".format(time.time() - current_time))

        # Extract the results of the benchmark
        stdout = benchmark.stdout.readlines()

        sample_timer = 0.1
        score_samples = []

        # Loop through lines until a line starts with "Elapsed time:" > 5 seconds
        for line in stdout:
            if line.decode("utf-8").startswith("Elapsed time:"):
                if float(line.decode("utf-8").split(" ")[2].strip()) > sample_timer:
                    sample_timer += 1
                    new_sample = line.decode("utf-8").split(" ")[6].strip()
                    if new_sample == "0.00":
                        new_sample = line.decode("utf-8").split(" ")[8].strip()
                    score_samples.append(int(float(new_sample)))

        print(score_samples)
        benchmark_runs.append(score_samples)
        # final_line = stdout[-1]
        # final_score = final_line.decode("utf-8").split(" ")[-1].strip()
        # run_results.append(final_score)


if do_elitism:
    current_parameter = "p"

    for index, generations in enumerate(elitism_range):
        current_time = time.time()

        # Run the r2v program with the specified image, adjusting the selected parameter over the specified range
        benchmark = subprocess.Popen([executable, "--outfile", "gen_output" + str(index) + ".svg", current_file_path, "-g", str(generations), "-s", str(default_size), "-p", str(default_elitism), "-t", str(default_time)], stdout=subprocess.PIPE)

        # Await the completion of the benchmark
        print("Benchmark started on parameter " + current_parameter + " = " + str(generations) + " (" + current_file_name + "): 0.00 seconds elapsed")
        while benchmark.poll() is None:
            # Stream the output of the benchmark
            # score_samples.append(benchmark.stdout.readlines()[-1].decode("utf-8").split(" ")[-1].strip())
            # print(benchmark.stdout.readlines()[-1])
            time.sleep(5)
            print("Benchmark running on parameter " + current_parameter + " = " + str(generations) + " (" + current_file_name + "): {:.2f} seconds elapsed".format(time.time() - current_time))

        # Extract the results of the benchmark


# Create line graph of the benchmark results
plot.figure(figsize=(8, 6))
# plot.plot(sample_times, benchmark_runs[0], marker='o', linestyle='-', color='b')
# plot.plot(sample_times, benchmark_runs[1], marker='o', linestyle='-', color='r')
# plot.ylim(0, None)
plot.yticks([0, 100, 300, 600, 1500, 5000, 12500, 25000, 50000])
# Set y axis distribution to ~logarithmic

# plot.yscale("log")
# plot.gca().invert_yaxis()
plot.plot(sample_times, benchmark_runs[0], linestyle='-', color='b')
plot.plot(sample_times, benchmark_runs[1], linestyle='-', color='r')
# plot.gca().set_yticklabels(1-plot.gca().get_yticks())

# plot.plot(sample_times, score_samples, marker='o', linestyle='-', color='r', label='y = x^2')

# Add title and labels
plot.title('Generation Size Benchmark')
plot.xlabel('Time (s)')
plot.ylabel('Mean-Squared Error')

# Add grid
plot.grid(True)

# Add legend
plot.legend()

# Save the plot
plot.savefig('size_benchmark.png', format='png')

plot.show()
