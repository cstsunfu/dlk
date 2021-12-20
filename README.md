# A Deep Learning Kit

This project is WIP, only provide some basic method, and is not been tested.

## What is this project do?

* Provide a templete for deep learning (especially for nlp) training and deploy.
* Provide parameters search.
* Provide basic architecture search.
* Provide some basic modules and models.
* Provide basic deploy method.

## Config example

## Will Complete

- [ ] Add more documents
- [ ] distill struct
- [ ] Add disable tokenizer post process

- [X] Predict

    - [X] make the main code.
    - [X] test

* [X] when the high config change the _name of the basic config, the base config should be coverd

- [ ] one Optimizer different Para Group use different Schedulers.
  
    - [ ] Ref [diff_schedule](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
    - [ ] add loss schedule, get best checkpoint by loss

- [ ] main entry? 

- [ ] LightGBM, it's necessary？
