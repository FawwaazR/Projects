# Final-Hate-Speech-Classifier
A Python class which provides a unified pipeline for:

- Cleaning and preprocessing raw text data.
    
- Performing topic modeling using one of three methods:

  - Latent Semantic Analysis (LSA),
      
  - Latent Dirichlet Allocation (LDA),

  - BERTopic.

- Classifying hate speech using a fine-tuned BERTweet model combined with a learning rate scheduler, 10% warmup, weight decay and 50% drop-out trained over 2 epochs of 1000 iterations each.

## Requirements 
Python 3.13

All libraries and packages from 'requirements.txt' file

It is recommended to create a virtual environment in which to install the above dependencies.

## Installation and Running
To generate the dashboard UI that produces the model results using the command line:

    1. From command line, go to the directory of the project folder (called code):

cd file_path\code

    2. Create a virtual environment in this directory (requires Python 3):

python -m venv environment_name

    3. Activate the virtual environment

environment_name\Scripts\activate

    4. Install required libraries and packages from 'requirements.txt' using pip:

pip install -r requirements.txt

    5. Run the code

python src\main.py

## Contributors
Main contributor: Fawwaaz Rusmaully

Supervisor: Dr Martyn Parker
