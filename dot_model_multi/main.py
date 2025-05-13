import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from unet3d import UNet3D
from simulate_mcx import simulate_mcx
from train import train_model
from visualize import visualize_voxel_output, visualize_slice_comparison

def setup_logging(log_dir: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('dot_model')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_dir is specified
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f'dot_model_{timestamp}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config: Dict[str, Any], save_dir: Path) -> None:
    """Save configuration to JSON file."""
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI-Enhanced Monte Carlo DOT: 3D Blood Oxygenation Reconstruction'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference'],
        default='train',
        help='Mode to run the model in'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint for inference'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save outputs'
    )
    return parser.parse_args()

def train(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run training pipeline."""
    try:
        # Create output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Save configuration
        save_config(config, run_dir)
        
        # Initialize model and training components
        logger.info("Initializing model and training components...")
        model = UNet3D(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1)
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        criterion = nn.MSELoss()
        
        # Train model
        logger.info("Starting training...")
        history = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.get('num_epochs', 10),
            batch_size=config.get('batch_size', 4),
            learning_rate=config.get('learning_rate', 1e-3),
            device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            save_dir=str(run_dir)
        )
        
        logger.info(f"Training completed! Final validation loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def inference(config: Dict[str, Any], checkpoint_path: str, logger: logging.Logger) -> None:
    """Run inference pipeline."""
    try:
        # Load model checkpoint
        logger.info(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model = UNet3D(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate test data
        logger.info("Generating test data...")
        test_data = simulate_mcx(
            voxel_dim=tuple(config.get('voxel_dim', (32, 32, 32))),
            activation_center=tuple(config.get('test_activation_center', (16, 16, 16))),
            activation_radius=config.get('test_activation_radius', 5)
        )
        
        # Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            output = model(test_data.unsqueeze(0))
        
        # Visualize results
        logger.info("Visualizing results...")
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save 3D visualization
        visualize_voxel_output(output[0])
        plt.savefig(output_dir / f'inference_3d_{timestamp}.png')
        plt.close()
        
        # Save slice comparison
        visualize_slice_comparison(
            test_data[0].unsqueeze(0),
            output[0],
            slice_idx=test_data.shape[1] // 2
        )
        plt.savefig(output_dir / f'inference_slices_{timestamp}.png')
        plt.close()
        
        logger.info("Inference completed!")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting DOT model...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Run appropriate pipeline
        if args.mode == 'train':
            train(config, logger)
        else:  # inference
            if args.checkpoint is None:
                raise ValueError("Checkpoint path must be provided for inference mode")
            inference(config, args.checkpoint, logger)
            
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("DOT model execution completed")

if __name__ == "__main__":
    main()