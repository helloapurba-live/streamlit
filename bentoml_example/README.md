# BentoML Iris Classifier Example

This is a simple example demonstrating how to use BentoML with scikit-learn and Pydantic.

## Prerequisites

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script to save the model to the BentoML model registry:

```bash
python train.py
```

## Serving the Model

Serve the model locally:

```bash
bentoml serve service.py:svc --reload
```

You can access the Swagger UI at `http://localhost:3000`.

## Building the Bento

Build the Bento for production:

```bash
bentoml build
```
