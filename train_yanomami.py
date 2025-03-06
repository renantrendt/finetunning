# Yanomami Translation Model Training Script
#
# This script serves as the entry point for training the Yanomami-English translation model
# It imports and uses the yanomami_trainer package modules

import logging
import sys
from yanomami_trainer.improvements_finetuning import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('yanomami_training.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Yanomami Translation Model Training")
    main()
    logger.info("Training completed successfully")
