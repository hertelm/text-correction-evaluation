# Text correction evaluation tool

This tool allows the evaluation of various text correction tasks, such as spelling correction, OCR post-correction or whitespace correction, with a standardized word-level F1 score as the metric.

## Installation

```commandline
git clone https://github.com/hertelm/text-correction-evaluation.git
cd gpt2-spell-checker
python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Run evaluations

The evaluation requires three files: a file with the misspelled sequences, a file with the predicted sequences, and a file with the ground truth.

`python3 main.py --misspelled <FILE> --predicted <FILE> --correct <FILE> -mp`

The argument `-mp` turns on multiprocessing.

To be able to see the results in the web app, specify an output path as follows:

`--out evaluation-webapp/results/<BENCHMARK>/<DEVELOPMENT/TEST>/<APPROACH_NAME>`

## Start the web app

```commandline
cd evaluation-webapp
python3 -m http.server <PORT>
```

The webapp is then accessible in the browser at `localhost:<PORT>/www`.

## Publication

When you use the evaluation tool in your work, please consider citing our upcoming publication:
*Matthias Hertel and Hannah Bast: "GPT-2-spell-checker: a tool for language-model-based spelling correction and evaluation" (2022, under review)*