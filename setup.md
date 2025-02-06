1. Setup `transformers` (tested with `python 370`):

```
python3.7 -m venv env_transformers
source env_transformers/bin/activate
pip install -U pip
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_transformers.txt
cd transformers
pip install --editable ./
```

2. Setup `TextAttack` library for evaluation. The version of the library I am using had some bugs for `python 3.7.0` at that point and therefore I recommend using `python 3.6.8`. 

```
python368 -m venv env_ta
source env_ta/bin/activate
pip install -U pip
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_textattack_368.txt
cd textAttack
pip install --editable ./
```

While running the `TextAttack` library, if you encounter the following error:
```
Traceback (most recent call last):
...
...
..., line 571, in __call__
decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
return self.data[item]
KeyError: 'labels
```
This is due to a bug in `TextAttack`, you can fix by changing the line 571 of the `data_collator.py` file in the transformers installation of your environment. 
As an example, change line 571 in file `env_ta/lib/python3.6/site-packages/transformers/data/data_collator.py`
from 

```
if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
```
to
```
if self.model is not None and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
```