import subprocess
import time
import matplotlib.pyplot as plot
# import numpy as np
from matplotlib.ticker import MaxNLocator

# Paths
executable = "../target/release/r2v"
discord_small = "../datasets/uvic-discord_small.jpg"
uvic_logo = "../datasets/uvic-logo.jpg"

# General variables
current_time = time.time()
current_file_name = "uvic-logo.jpg"
current_file_path = uvic_logo
current_parameter = "g"

default_generations = 500
default_size = 1000
default_time = 30
default_elitism = 0.05

run_results = []
score_samples = []
sample_times = []
benchmark_runs = []

# Benchmark variables
generations_range = [250, 500, 750, 1000, 1250]
size_range = [500, 1000, 1500, 2000, 2500]
elitism_range = [0.01, 0.05, 0.1, 0.15, 0.2]
line_colours = ['b', 'r', 'g', 'y', 'm']

# Create benchmark to organise results
class Benchmark:
    def __init__(self):
        self.size_range = size_range
        self.elitism_range = elitism_range
        self.sample_times = sample_times
        self.benchmark_runs = benchmark_runs
        self.line_colours = line_colours
        self.labels = []

size_benchmark = Benchmark()
size_benchmark.labels = ["500", "1000", "1500", "2000", "2500"]

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
        benchmark = subprocess.Popen([executable, "--outfile", "gen_output" + str(index) + ".svg", current_file_path, "-g", str(generations), "-s", str(default_size), "-p", str(default_elitism), "-t", "600"], stdout=subprocess.PIPE)

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
            time.sleep(5)
            print("Benchmark running on parameter " + current_parameter + " = " + str(size) + " (" + current_file_name + "): {:.2f} seconds elapsed".format(time.time() - current_time))

        # Extract the results of the benchmark
        stdout = benchmark.stdout.readlines()

        sample_delay = 1
        eval_line = ""
        score_samples = []
        sample_times_list = []

        # Loop through lines
        for line in stdout:
            if line.decode("utf-8").startswith("Elapsed time:"):
                sample_time = float(line.decode("utf-8").split(" ")[2].strip())
                if float(sample_time > sample_delay):
                    new_sample = eval_line.decode("utf-8").split(" ")[1].strip()
                    sample_times_list.append(sample_time)
                    score_samples.append(int(float(new_sample)))
            else:
                eval_line = line
                
        # print(score_samples)
        benchmark_runs.append(score_samples)
        sample_times.append(sample_times_list)

    # Create line graph of the benchmark results
    plot.figure(figsize=(8, 6))

    for i in range(len(benchmark_runs)):
        plot.plot(sample_times[i], benchmark_runs[i], linestyle='-', color=line_colours[i], label=size_benchmark.labels[i])

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
