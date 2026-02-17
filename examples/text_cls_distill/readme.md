#### Step 1: 启动TeacherAPI
打开一个终端，运行 Mock 服务：
```bash
python mock_teacher_server.py
```

#### Step 2: 离线抓取蒸馏数据（Preprocessing）
打开第二个终端，运行数据处理脚本：
```bash
python process.py
```

#### Step 3: 启动子模型蒸馏训练
当数据预处理完毕并保存到 `data/processed_data` 后，即可执行训练：
```bash
python train.py
```
