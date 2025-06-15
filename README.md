# ACM Workshop on Systems for ML (ACM_WS_SYSML)

A federated learning implementation for energy-efficient distributed training.

## Repository Structure

```
ACM_WS_SYSML/
├── configs/              # Configuration files
│   └── config.yaml      # Main configuration file
├── models/              # Model implementations
│   └── CNN.py          # CNN model implementation
├── initial_models/      # Pre-trained model checkpoints
├── client.py           # Client implementation
├── server.py           # Server implementation
├── run_exp.py          # Main experiment runner
└── requirements.txt    # Project dependencies
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download data
```bash
wget https://www.dropbox.com/scl/fi/3bqhiz8uzcol2ge28ruce/data.zip?rlkey=ljhilzs8qam2m0mohru3otmx7&st=wkc7ehq7&dl=0 -O data.zip
unzip data.zip  
```

4. Run an experiment:
```bash
python run_exp.py --config configs/config.yaml
```

## Configuration

The `config.yaml` file contains experiment parameters including:
- Model architecture settings
- Training parameters
- Client/server configuration
- Dataset settings

## Models

Currently supported models:
- CNN (Convolutional Neural Network)

## License

This project is licensed under the MIT License.
