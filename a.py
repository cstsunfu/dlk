from dlkit.utils.logger import setting_logger
setting_logger("./test.log")
from dlkit.train import Train


trainer = Train('./tasks/test_cls.hjson')
trainer.run()

# {
    # "root": {
        # "task": {
            # "datamodule": {
                # "config": {
                    # "pin_memory": null,
                    # "collate_fn": "default",
                    # "shuffle": {
                        # "train": true,
                        # "predict": false,
                        # "valid": false,
                        # "test": false,
                        # "online": false
                    # },
                    # "key_type_pairs": {
                        # "input_ids": "int"
                    # },
                    # "gen_mask": {
                        # "input_ids": "attention_mask"
                    # },
                    # "key_padding_pairs": {
                        # "input_ids": 0
                    # },
                    # "train_batch_size": 32,
                    # "predict_batch_size": 32,
                    # "online_batch_size": 1
                # },
                # "_name": "basic"
            # },
            # "imodel": {
                # "config": {},
                # "schedule": {
                    # "config": {
                        # "schedule_name": "linear_warmup,",
                        # "schedule_config": {
                            # "num_warmup_steps": 0.1
                        # }
                    # },
                    # "_name": "basic"
                # },
                # "optimizer": {
                    # "config": {
                        # "optimizer_name": "adamw,",
                        # "optimizer_config": {
                            # "lr": 0.001,
                            # "betas": [
                                # 0.9,
                                # 0.999
                            # ],
                            # "eps": 1e-06,
                            # "weight_decay": 0.01
                        # },
                        # "optimizer_special_groups": [
                            # [
                                # "bias & LayerNorm.bias & LayerNorm.weight",
                                # {
                                    # "weight_decay": 0
                                # }
                            # ]
                        # ]
                    # },
                    # "_name": "basic"
                # },
                # "loss": {
                    # "config": {
                        # "task_name": "classification",
                        # "weight": null,
                        # "ignore_index": -1,
                        # "label_smoothing": 0,
                        # "pred_truth_pair": [],
                        # "loss_scale": []
                    # },
                    # "_name": "cross_entropy"
                # },
                # "model": {
                    # "config": {
                        # "embedding_dim": 50,
                        # "dropout": 0.1,
                        # "embedding_file": "test_cls/processed_data.pkl",
                        # "embedding_trace": "embedding"
                    # },
                    # "initmethod": {
                        # "config": {
                            # "range": 0.01
                        # },
                        # "_name": "range_norm,"
                    # },
                    # "encoder": {
                        # "config": {
                            # "return_logits": "encoder_logits",
                            # "output_map": {},
                            # "hidden_size": 50,
                            # "input_size": 50,
                            # "output_size": 50,
                            # "num_layers": 1,
                            # "dropout": 0.1
                        # },
                        # "module": {
                            # "config": {
                                # "bidirectional": true,
                                # "hidden_size": 50,
                                # "input_size": 50,
                                # "proj_size": 50,
                                # "num_layers": 1,
                                # "dropout": 0.1,
                                # "dropout_last": true
                            # },
                            # "_name": "lstm"
                        # },
                        # "_name": "lstm"
                    # },
                    # "decoder": {
                        # "config": {
                            # "input_size": 50,
                            # "output_size": 50,
                            # "pool": null,
                            # "dropout": 0.1,
                            # "return_logits": "decoder_logits",
                            # "output_map": {}
                        # },
                        # "module": {
                            # "config": {
                                # "input_size": 50,
                                # "output_size": 50,
                                # "dropout": 0.1,
                                # "pool": null
                            # },
                            # "_name": "linear"
                        # },
                        # "_name": "linear"
                    # },
                    # "embedding": {
                        # "config": {
                            # "embedding_file": "test_cls/processed_data.pkl",
                            # "embedding_dim": 50,
                            # "embedding_trace": "embedding",
                            # "freeze": false,
                            # "dropout": 0.1,
                            # "output_map": {},
                            # "return_logits": "embedding_logits"
                        # },
                        # "_name": "static"
                    # },
                    # "_name": "basic"
                # },
                # "_name": "basic"
            # },
            # "_name": "test_cls"
        # },
        # "config": {
            # "save_dir": "test_cls",
            # "data_path": "test_cls/processed_data.pkl"
        # },
        # "_name": "test_cls"
    # }
# }




  # {
      # "root": {
          # "task": {
              # "datamodule": {
                  # "config": {
                      # "pin_memory": null,
                      # "collate_fn": "default",
                      # "shuffle": {
                          # "train": true,
                          # "predict": false,
                          # "valid": false,
                          # "test": false,
                          # "online": false
                      # },
                      # "key_type_pairs": {
                          # "input_ids": "int"
                      # },
                      # "gen_mask": {
                          # "input_ids": "attention_mask"
                      # },
                      # "key_padding_pairs": {
                          # "input_ids": 0
                      # },
                      # "train_batch_size": 32,
                      # "predict_batch_size": 32,
                      # "online_batch_size": 1
                  # },
                  # "_name": "basic"
              # },
              # "imodel": {
                  # "config": {},
                  # "schedule": {
                      # "config": {
                          # "schedule_name": "linear_warmup,",
                          # "schedule_config": {
                              # "num_warmup_steps": 0.1
                          # }
                      # },
                      # "_name": "basic"
                  # },
                  # "optimizer": {
                      # "config": {
                          # "optimizer_name": "adamw,",
                          # "optimizer_config": {
                              # "lr": 0.001,
                              # "betas": [
                                  # 0.9,
                                  # 0.999
                              # ],
                              # "eps": 1e-06,
                              # "weight_decay": 0.01
                          # },
                          # "optimizer_special_groups": [
                              # [
                                  # "bias & LayerNorm.bias & LayerNorm.weight",
                                  # {
                                      # "weight_decay": 0
                                  # }
                              # ]
                          # ]
                      # },
                      # "_name": "basic"
                  # },
                  # "loss": {
                      # "config": {
                          # "task_name": "classification",
                          # "weight": null,
                          # "ignore_index": -1,
                          # "label_smoothing": 0,
                          # "pred_truth_pair": [],
                          # "loss_scale": []
                      # },
                      # "_name": "cross_entropy"
                  # },
                  # "model": {
                      # "config": {
                          # "embedding_dim": 50,
                          # "dropout": 0.1,
                          # "embedding_file": "test_cls/processed_data.pkl",
                          # "embedding_trace": "embedding"
                      # },
                      # "initmethod": {
                          # "config": {
                              # "range": 0.01
                          # },
                          # "_name": "range_norm,"
                      # },
                      # "encoder": {
                          # "config": {
                              # "return_logits": "encoder_logits",
                              # "output_map": {},
                              # "hidden_size": 50,
                              # "input_size": 50,
                              # "output_size": 50,
                              # "num_layers": 1,
                              # "dropout": 0.1
                          # },
                          # "module": {
                              # "config": {
                                  # "bidirectional": true,
                                  # "hidden_size": 50,
                                  # "input_size": 50,
                                  # "proj_size": 50,
                                  # "num_layers": 1,
                                  # "dropout": 0.1,
                                  # "dropout_last": true
                              # },
                              # "_name": "lstm"
                          # },
                          # "_name": "lstm"
                      # },
                      # "decoder": {
                          # "config": {
                              # "input_size": 50,
                              # "output_size": 50,
                              # "pool": null,
                              # "dropout": 0.1,
                              # "return_logits": "decoder_logits",
                              # "output_map": {}
                          # },
                          # "module": {
                              # "config": {
                                  # "input_size": 50,
                                  # "output_size": 50,
                                  # "dropout": 0.1,
                                  # "pool": null
                              # },
                              # "_name": "linear"
                          # },
                          # "_name": "linear"
                      # },
                      # "embedding": {
                          # "config": {
                              # "embedding_file": "test_cls/processed_data.pkl",
                              # "embedding_dim": 50,
                              # "embedding_trace": "embedding",
                              # "freeze": false,
                              # "dropout": 0.1,
                              # "output_map": {},
                              # "return_logits": "embedding_logits"
                          # },
                          # "_name": "static"
                      # },
                      # "_name": "basic"
                  # },
                  # "_name": "basic"
              # },
              # "manager": {
                  # "config": {
                      # "callbacks": [],
                      # "logger": true,
                      # "enable_checkpointing": false,
                      # "accelerator": null,
                      # "default_root_dir": null,
                      # "gradient_clip_val": null,
                      # "gradient_clip_algorithm": null,
                      # "num_nodes": 1,
                      # "num_processes": 1,
                      # "devices": null,
                      # "gpus": null,
                      # "auto_select_gpus": false,
                      # "tpu_cores": null,
                      # "ipus": null,
                      # "log_gpu_memory": null,
                      # "enable_progress_bar": true,
                      # "overfit_batches": 0,
                      # "track_grad_norm": -1,
                      # "check_val_every_n_epoch": 1,
                      # "fast_dev_run": false,
                      # "accumulate_grad_batches": 1,
                      # "max_epochs": null,
                      # "min_epochs": null,
                      # "max_steps": -1,
                      # "min_steps": null,
                      # "max_time": null,
                      # "limit_train_batches": 1,
                      # "limit_val_batches": 1,
                      # "limit_test_batches": 1,
                      # "limit_predict_batches": 1,
                      # "val_check_interval": 1,
                      # "log_every_n_steps": 50,
                      # "strategy": "ddp",
                      # "sync_batchnorm": false,
                      # "precision": 32,
                      # "enable_model_summary": true,
                      # "weights_summary": "top",
                      # "weights_save_path": null,
                      # "num_sanity_val_steps": 2,
                      # "resume_from_checkpoint": null,
                      # "profiler": null,
                      # "benchmark": false,
                      # "deterministic": false,
                      # "reload_dataloaders_every_n_epochs": 0,
                      # "auto_lr_find": false,
                      # "replace_sampler_ddp": true,
                      # "detect_anomaly": false,
                      # "auto_scale_batch_size": false,
                      # "plugins": null,
                      # "amp_backend": "native",
                      # "amp_level": null,
                      # "move_metrics_to_cpu": false,
                      # "multiple_trainloader_mode": "max_size_cycle",
                      # "stochastic_weight_avg": false,
                      # "terminate_on_nan": null
                  # },
                  # "_name": "lightning"
              # },
              # "_name": "test_cls"
          # },
          # "config": {
              # "save_dir": "test_cls",
              # "data_path": "test_cls/processed_data.pkl"
          # },
          # "_name": "test_cls"
      # }
  # }

