# A Deep Learning ToolKit

This project is WIP.

[Read the Docs](https://dlk.readthedocs.io/en/latest/)


## Install

```
pip install dlk

or 
git clone this repo and do

python setup.py install

```
## What's this?

* Provide a templete for deep learning (especially for nlp) training and deploy.
* Provide parameters search.
* Provide basic architecture search.
* Provide some basic modules and models.
* Provide reuse the pretrained model for predict.

## More Feature is Comming


* [ ] Tasks support
    * [ ] NLP
        * [X] Classification 
        * [X] Pair Classification 
        * [X] Regression 
        * [X] Pair Regression 
        * [X] Sequence Labeling
        * [X] Span Classification
        * [X] Relation Extraction
        * [X] Token Rerank
        * [ ] MRC SQuAD
        * [ ] Translation
        * [ ] Summary
    * [ ] CV
        * [ ] Classification 

- [ ] Generate models.

- [ ] Distill structure.

- [ ] Ensemble models for NLU(and check how to do this in NLG)

- [ ] Training Strategy
    - [X] Adversarial Training(FGM/PGD/FreeLB)
    - [X] Schedule Loss(you can control the loss schedule)
    - [X] Schedule MultiTask Loss(you can control the loss schedule for each task)
    - [X] Focal Loss

- [ ] Online service by triton.

- [ ] Data Augment.

- [ ] ~~Support LightGBM. Will split to another package.~~

* [ ] Make most complexity modules like Beam Search, CRF to be scriptable.

* [X] Add UnitTest
    * [X] Parser
    * [X] Tokenizer
    * [X] Config
    * [X] Link
