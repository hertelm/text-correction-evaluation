import os
import time
import argparse
import multiprocessing as mp
import json

from helper.files import read_lines, read_file
from evaluation.evaluator import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the predictions of a spell checking program on a given"
                                                 " benchmark of misspelled and correct text.")
    parser.add_argument("--correct", required=True, type=str,
                        help="Path to the file with the ground truth sequences.")
    parser.add_argument("--misspelled", required=True, type=str,
                        help="Path to the file with the misspelled sequences.")
    parser.add_argument("--predictions", required=True, type=str,
                        help="Path to the file with the predicted sequences.")
    parser.add_argument("--words", type=str, default="data/words.txt",
                        help="Path to the vocabulary (one word per line, default: data/words.txt).")
    parser.add_argument("-n", type=int, default=None,
                        help="Number of sequences to evaluate (default: all).")
    parser.add_argument("-mp", action="store_true",
                        help="Use multiprocessing.")
    parser.add_argument("--out", required=False, type=str,
                        help="Evaluation output file (file endings will be attached).")
    args = parser.parse_args()

    words = set(read_lines(args.words))
    print(len(words))

    evaluator = Evaluator(words)

    start = time.monotonic()

    correct_sequences = read_file(args.correct)[:args.n]
    corrupt_sequences = read_file(args.misspelled)[:args.n]
    predicted_sequences = read_file(args.predictions)[:args.n]

    n_cpus = mp.cpu_count() if args.mp else 1

    with mp.Pool(n_cpus) as pool:
        results = pool.starmap(evaluator.evaluate_sample,
                               list(zip(correct_sequences, corrupt_sequences, predicted_sequences)))

    for evaluations, is_correct, _ in results:
        for labels, case, error_type in evaluations:
            evaluator.add(labels, case, error_type)
        evaluator.add_sequence_result(is_correct)

    if args.out is not None:
        sequences_dir = args.out + ".sequences/"
        if not os.path.exists(sequences_dir):
            os.mkdir(sequences_dir)
        for i, (_, _, evaluated_sequence) in enumerate(results):
            sequence_file = sequences_dir + "%i.json" % i
            with open(sequence_file, "w") as f:
                f.write(evaluated_sequence.to_json() + "\n")
        results_dict = evaluator.get_results_dict()
        results_file = args.out + ".results.json"
        with open(results_file, "w") as f:
            f.write(json.dumps(results_dict))

    evaluator.print_evaluation()
    end = time.monotonic()
    print("%.1f%% sequence accuracy (%i/%i)" % (evaluator.sequence_accuracy() * 100,
                                                evaluator.n_correct_sequences,
                                                evaluator.n_sequences))
    print()
    print(f"Processing {evaluator.n_sequences} sequences took {end - start:.2f} seconds")
