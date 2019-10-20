# philo

>> Bringing modalities together since eternity

*Philo* (derived from [Philophrosyne](https://en.wikipedia.org/wiki/Philophrosyne), the goddess of friendship) is a module for dynamic fusion of multiple modalities. It possess two special spells right now, namely, *transFusion* and *ganFusion*. Both of them fall under high-level fusion-magic class of *TrGF* (pronounced *trijfu*). Philo's *src* of power is PyTorch.


Leave a :star: if you find the code useful :smiley:


## Usage instructions:

Make sure that you are in the root directory before running the following commands:
* Modify the `src/models/config.py` file appropriately before running any commands
* If it's your first run (starting with nothing but the raw files), run `python src/process_raw.py`. This will create processed dataset files inside `data/processed`
* Once you have the processed files, run `python src/train.py` to train the network. It will take care of:
    - Preprocessing the text
    - Creating/Loading vocabulary
    - Generating/Loading embeddings
    - Training the model
* Run `python src/predict.py` to test the trained model
