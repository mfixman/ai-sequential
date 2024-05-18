# AI-sequential

## Introduction
This repository contains the code for our deep learning project focused on abstractive text summarisation.

## Installation
To replicate the environment and run the code follow these steps (**Warning**: a CUDA device must be used):

1. Clone the repository to your local machine 

    ```bash
    git clone https://github.com/mfixman/ai-sequential
    cd ai-sequential
    ```

2. Install dependencies

    - Using pip
        ```bash
        pip install -r requirements.txt
        ```
      
    - WandB:
   
         To setup WandB logging, you must set the WandB key as an environment variable. An invitattion link to the wandB log (NewSum) is provided here: https://wandb.ai/aabati?invited=&newUser=false

## Usage

1. Dataset

    To create the .pickle files use from the dataset class. Run the 'dataset_tokenizer.py' file:
    ```bash
    python dataset_tokenizer.py
    ```

        The script will tokenize and preprocess the CNN/Daily Mail dataset from HuggingFace.

2. Train the models

    To train the model, run:
    ```bash
    python train.py
    ```
    
    To specify the parameters you can either:

        1. Change the 'model_params' and 'train' settings in the 'config.yaml' file.

            For example:

            ```yaml
            data_settings:
                dataset_path: 'tokenized_data_subword_v2'
                vocabulary_path: 'tokenized_data_subword_v2/vocabulary.pickle'
                special_tokens: ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            model_params:
                model_name: 'transformer'
                version: '2'
                encoder_embedding_dim: 512
                decoder_embedding_dim: 512
                hidden_dim: 512
                num_layers: 6
                dropout: 0.2
            train:
                epochs: 5
                batch_size: 2
                learning_rate: 0.0001
                varkappa: 0
                checkpoint_folder: 'checkpoints'
                load_checkpoint : False
                log: False
                max_samples: 10
            infer:
                batch_size: 1
                checkpoint_folder: 'checkpoints'
                load_checkpoint : True
            ```

        2. Use the parameters directly from the command line.

            For example:
            ```bash
            python train.py --model_name transformer --max_samples 10
            ```

3. Inference