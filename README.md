```
This project is WIP, only provide some basic method, and is not been tested.
```
## What is this project?

* Provide a templete for deep learning (especially for nlp) training and deploy.
* Provide parameters search.
* Provide basic architecture search.
* Provide some basic modules and models.
* Provide basic deploy method.

## Config example

WIP

## TODAY TODO
- [ ] Add token gather module for ner
- [ ] refactor the postprocess for first piece
- *add char embedding by cnn

## TODO

- [ ] Add documents
- [ ] distill struct and *api* setting
- [ ] Add disable tokenizer post process
- [o] Predict
  - [X] make the main code.
  - [ ] test
* [X] when the high config change the _name of the basic config, the base config should be coverd
- [ ] one Optimizer different Para Group use different Schedulers.
  
  - [ ] Ref  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
  - [ ] add loss schedule, 
     - [ ] how to get best checkpoint by loss

- [ ] main entry? 
  - [ ] how to apply


- [ ] LightGBM
 - [ ]  It's necessary？

## NOTE
