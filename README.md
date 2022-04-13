
# A Deep Learning ToolKit

This project is WIP.

[Read the Docs](https://dlk.readthedocs.io/en/latest/)


## Install

```
pip install dlk

or 
git clong this repo and do

python setup.py install

```
## What is this project do?

* Provide a templete for deep learning (especially for nlp) training and deploy.
* Provide parameters search.
* Provide basic architecture search.
* Provide some basic modules and models.
* Provide basic deploy method.

## More Feature is Comming

- [ ] Add more documents.
- [ ] Distill structure.
- [ ] Adv training.
- [ ] Add disable tokenizer post process.

- [ ] Predict
    - [X] Complete the main predict code.
    - [ ] Test.
    - [ ] Convert to TorchScript.
    - [ ] Convert to ONNXRT

* [X] When the high config change the _name of the basic config, the base config should be coverd.

- [ ] One `optimizer` different para groups use different `scheduler`s.
  
    - [ ] Ref [diff_schedule](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
    - [ ] Add loss schedule, get best checkpoint by loss.

- [ ] Support LightGBM, it's maybe not necessary? Will split to another package.
* [ ] Make CRF and more modules which uses the op like `for` and `if` to be scriptable（`for`，`if`）
* [ ] Validating LSTM module is scriptable or not?
* [ ] Add UnitTest
