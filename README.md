
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
* Provide reuse the pretrained model for predict.

## More Feature is Comming


- [ ] Generate models.

- [ ] Distill structure.

- [ ] Computer vision support.

- [ ] Online service
    - [ ] Provide a web server for online predict.

- [ ] One `optimizer` different para groups use different `scheduler`s. [diff_schedule](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
~~- [ ] Support LightGBM, it's maybe not necessary? Will split to another package.~~
* [ ] Make most modules like CRF to be scriptable

* [X] Add UnitTest
    * [X] Parser
    * [X] Tokenizer
    * [X] Config
    * [X] Link
