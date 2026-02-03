import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisOutput(BaseModel):
    prediction: int

# Load the runner
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Create the service
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=JSON(pydantic_model=IrisInput), output=JSON(pydantic_model=IrisOutput))
def classify(input_data: IrisInput) -> IrisOutput:
    input_df = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]
    result = iris_clf_runner.predict.run(input_df)
    return IrisOutput(prediction=result[0])
