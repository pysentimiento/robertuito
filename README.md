# RoBERTuito
## A pre-trained language model for social media text in Spanish

[**READ THE FULL PAPER**](https://arxiv.org/abs/2111.09453)
[Github Repository](https://github.com/pysentimiento/robertuito)

*RoBERTuito* is a pre-trained language model for user-generated content in Spanish, trained following RoBERTa guidelines on 500 million tweets. *RoBERTuito* comes in 3 flavors: cased, uncased, and uncased+deaccented.

We tested *RoBERTuito* on a benchmark of tasks involving user-generated text in Spanish. It outperforms other pre-trained language models for this language such as *BETO*, *BERTin* and *RoBERTa-BNE*. The 4 tasks selected for evaluation were: Hate Speech Detection (using SemEval 2019 Task 5, HatEval dataset), Sentiment and Emotion Analysis (using TASS 2020 datasets), and Irony detection (using IrosVa 2019 dataset).

| model              | hate speech     | sentiment analysis   | emotion analysis   | irony detection  |   score |
|:-------------------|:----------------|:---------------------|:-------------------|:-----------------|---------:|
| robertuito-uncased | 0.801 ± 0.010  | 0.707 ± 0.004       | 0.551 ± 0.011     | 0.736 ± 0.008   | 0.6987  |
| robertuito-deacc   | 0.798 ± 0.008  | 0.702 ± 0.004       | 0.543 ± 0.015     | 0.740 ± 0.006   | 0.6958  |
| robertuito-cased   | 0.790 ± 0.012  | 0.701 ± 0.012       | 0.519 ± 0.032     | 0.719 ± 0.023   | 0.6822  |
| roberta-bne        | 0.766 ± 0.015  | 0.669 ± 0.006       | 0.533 ± 0.011     | 0.723 ± 0.017   | 0.6726  |
| bertin             | 0.767 ± 0.005  | 0.665 ± 0.003       | 0.518 ± 0.012     | 0.716 ± 0.008   | 0.6666  |
| beto-cased         | 0.768 ± 0.012  | 0.665 ± 0.004       | 0.521 ± 0.012     | 0.706 ± 0.007   | 0.6651  |
| beto-uncased       | 0.757 ± 0.012  | 0.649 ± 0.005       | 0.521 ± 0.006     | 0.702 ± 0.008   | 0.6571  |


We release the pre-trained models on huggingface model hub:

- [RoBERTuito uncased](https://huggingface.co/pysentimiento/robertuito-base-uncased)
- [RoBERTuito cased](https://huggingface.co/pysentimiento/robertuito-base-cased)
- [RoBERTuito deacc](https://huggingface.co/pysentimiento/robertuito-base-deacc)

## Usage

**IMPORTANT -- READ THIS FIRST**

*RoBERTuito* is not yet fully-integrated into `huggingface/transformers`. To use it, first install `pysentimiento`

```bash
pip install pysentimiento
```

and preprocess text using `pysentimiento.preprocessing.preprocess_tweet` before feeding it into the tokenizer

```python
from transformers import AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet

tokenizer = AutoTokenizer.from_pretrained('pysentimiento/robertuito-base-cased')

text = "Esto es un tweet estoy usando #Robertuito @pysentimiento 🤣"
preprocessed_text = preprocess_tweet(text, ha)

tokenizer.tokenize(preprocessed_text)
# ['<s>','▁Esto','▁es','▁un','▁tweet','▁estoy','▁usando','▁','▁hashtag','▁','▁ro','bert','uito','▁@usuario','▁','▁emoji','▁cara','▁revolviéndose','▁de','▁la','▁risa','▁emoji','</s>']
```

We are working on integrating this preprocessing step into a Tokenizer within `transformers` library

## Development

### Installing

We use `python==3.7` and `poetry` to manage dependencies.

```bash
pip install poetry
poetry install
```

### Benchmarking

To run benchmarks

```bash
python bin/run_benchmark.py <model_name> --times 5 --output_path <output_path>
```

Check [RUN_BENCHMARKS](RUN_BENCHMARKS.md) for all experiments

### Smoke test
Test the benchmark running

```
./smoke_test.sh
```
## Citation

If you use *RoBERTuito*, please cite our paper:

```bibtex
@misc{perez2021robertuito,
      title={RoBERTuito: a pre-trained language model for social media text in Spanish},
      author={Juan Manuel Pérez and Damián A. Furman and Laura Alonso Alemany and Franco Luque},
      year={2021},
      eprint={2111.09453},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```