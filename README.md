# Code to run experiments from the ACL 23 paper Don't Retrain, Just Rewrite: Countering Adversarial Perturbations by Rewriting Text 

## Setup the code

Please refer to `setup.md` to setup the virtual environments.

## Run the code

1. In order to train a rewriter based model, we need to first extract the adversarial examples for one of the adversarial attacks. You can use the same set of adversarial examples that you might use for adversarial training. With the `textattack` toolkit, you can use the following command to generate adversarial examples for the training set. I will use the commands for the `sst-2` dataset as an example:

```
textattack attack --recipe textfooler \
                --model-from-huggingface textattack/bert-base-uncased-SST-2 \
                --dataset-from-huggingface glue^sst2 \
                --num-examples -1 --dataset-split train \
                --log-to-csv textfooler_adv_train.csv
```
This saves adversarial examples to a csv file. Setting `--num-examples` to -1 generates adversarial samples for all examples from the training set.

2. Since the defense model is an encoder-decoder model, we need to convert the adversarial examples in the `textfooler_adv_train.csv` generated above to the required text-in, text-out format. You can use `scripts/reformat_textattack_file_to_t5_training_multiple.py` for this.

```
python scripts/reformat_textattack_file_to_t5_training_multiple.py textfooler_adv_train.csv textfooler_adv_train_reformatted.csv 
```

Please refer to the sample files in `sample_files/` to see the format of these files.

3. Train the rewriter model using the `transformers` library. Run the `run_summarization.py` script:

```
python run_summarization.py --model_name_or_path t5-base --do_train --do_eval \
                --train_file textfooler_adv_train_reformatted.csv \
                --validation_file textfooler_adv_train_reformatted.csv \
                --source_prefix "correct the given sentence: "  \
                --output_dir sst2_rewriter --overwrite_output_dir  \
                --per_device_train_batch_size=16 \
                --predict_with_generate --per_device_eval_batch_size=16 \
                --text_column adversarial --summary_column original \
                --save_steps 10000000 --num_train_epochs 5
```

4. Run evaluation using `TextAttack`. Go to the `textAttack` folder and activate its virtual environment and run the following command for evaluation:

```
textattack attack --recipe textfooler \
                --model-from-huggingface textattack/bert-base-uncased-SST-2 \
                --dataset-from-huggingface glue^sst2 --num-examples 100 \
                --dataset-split validation  --log-to-csv sst2_with_writer.csv \
                --model-type-to-evaluate t5 \
                --t5-model-to-use ../transformers/examples/pytorch/summarization/sst2_rewriter \
                --other-args '{"generate_sentence_by_sentence": "false", "generation_method":"full", "task_name":"sst2", "num_labels":"2", "source_prefix":"correct the given sentence: "}'
```

If you have a different classification model, say `ag_news`, change `--other-args` to reflect that, example:
```
--other-args '{"generate_sentence_by_sentence": "false", "generation_method":"full", "task_name":"ag_news", "num_labels":"4", "source_prefix":"correct the given sentence: "}'
```

Also, if you set `generate_sentence_by_sentence` to `true`, this will enable the model to generate the output sentence-by-sentence for a multi-sentence input text (say, paragraph text for `imdb` or `ag_news`).