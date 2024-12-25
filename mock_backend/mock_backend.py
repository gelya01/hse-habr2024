from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Tuple, Any

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

models = {}

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    hyperparameters: Dict[str, Any]

class FitRequest(BaseModel):
    model_info: ModelInfo
    text: List[str]
    labels: List[List[str]]

class FitResponse(BaseModel):
    message: str

class PredictRequest(BaseModel):
    model_id: str
    text: str

class PredictResponse(BaseModel):
    labels: List[str]


@app.post("/fit")
async def fit(requests: List[FitRequest]) -> List[FitResponse]:
    responses = []
    for request in requests:
        model_trained = None # нужно добавить код для обучения моделей
        models[request.model_info.model_id] = {
            "model_info": request.model_info,
            "model": model_trained
        }
        responses.append(FitResponse(message=f"model {request.model_info.model_id} is fit"))
        
    return responses

@app.post("/predict")
async def predict(requests: List[PredictRequest]) -> List[PredictResponse]:
    responses = []
    for request in requests:
        model_id = request.model_id
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        responses.append(PredictResponse(labels=["mocked_label_1", "mocked_label_2"]))
    return responses


@app.get("/list_models")
async def list_models() -> List[ModelInfo]:
    return [models[model]["model_info"] for model in models]


@app.get("/models/{model_id}")
async def get_model(model_id: str) -> List[ModelInfo]:
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    return [models[model_id]["model_info"]]


if __name__ == "__main__":
    uvicorn.run("mock_backend:app", host="0.0.0.0", port=8000, reload=True)