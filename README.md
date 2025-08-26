# AutoML Service: A Case Study

This project is a case study created for an interview to demonstrate the architectural thinking behind Agentic automl system.

**Disclaimer:** Due to time constraints,
I have just built demonstration for an "Agentic AutoML" service and how analysis module can be integrated with agents.The logic within individual files is kept minimal and not so clean. Most of the files just have placeholders and serve mainly to demonstrate the intended architecture and workflow.

model training logic can be found in ```core/services/tools/train.py``` file. all the codes 

## Project Overview


## Project Structure

The repository is organized into the following main directories:

-   `core/`: Contains the core microservices of the application.
    -   `agent_service/`: An agentic docker API service that orchestrates the AutoML pipeline.
    -   `model_service/`: A docker API service that provides individual machine learning functionalities.
-   `data/`: Intended for storing datasets used by the AutoML service.
-   `deployment/`: Contains deployment configurations, such as `docker-compose.yaml` and environment files.

## The "Agentic AutoML" Concept

The core idea of this project is to use an agent to drive the AutoML process. This is implemented in the `agent_service` using `langgraph`.

The agent follows a predefined graph of operations to:

1. Identifies Data distribution and comes up with model config
2.  Preprocess the data.
3.  Extract features.
4.  Train multiple clustering algorithms.
5.  Select the best performing algorithm based on a given metric.
6.  Register the winning model.

This agent-based approach allows for a more flexible and extensible AutoML pipeline, where new steps or algorithms can be easily added to the agent's workflow.

## Services

### Model Service (`core/model_service`)

This service acts as a set of building blocks for our AutoML system. It exposes a simple API for performing individual machine learning tasks, such as:

-   Data preprocessing
-   Feature extraction
-   Model training
-   Model selection

### Agent Service (`core/agent_service`)

This is the main orchestrator of the AutoML pipeline. It takes a high-level job specification from the user and uses the `model_service` to execute the steps defined in the `agent_graph`.

## How to Run

The services are designed to be run using Docker Compose. You can start the entire system by running:

```bash
docker-compose up
```

This will build and start the `agent_service` and the `model_service`.

## Conclusion

This project provides a high-level architectural blueprint for an Agentic AutoML system. While the implementation details are not fully fleshed out, it successfully demonstrates a modern approach to building complex, agent-driven machine learning systems with a clean, microservices-based architecture.
