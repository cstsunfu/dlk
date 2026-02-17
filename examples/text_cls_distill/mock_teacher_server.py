# mock_teacher_server.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictRequest(BaseModel):
    sentence: str


@app.post("/predict/teacher")
async def predict_teacher(req: PredictRequest):
    """
    Mock Teacher API.
    根据句子中是否包含特定积极词汇，返回模拟的 logits。
    """
    text = req.sentence.lower()

    # 模拟模型推理 (Class 0: Negative, Class 1: Positive)
    if "good" in text or "great" in text:
        logits = [-2.5, 3.1]  # 强烈的正向置信度
    elif "bad" in text or "terrible" in text:
        logits = [2.8, -1.9]  # 强烈的负向置信度
    else:
        logits = [0.1, -0.1]  # 模棱两可

    return {"status": "success", "result": {"logits": logits}}


if __name__ == "__main__":
    # 启动服务在 8080 端口
    uvicorn.run(app, host="127.0.0.1", port=8080)
