# Repository to fetch and process data from the [RUV](https://www.ruv.is/x21/) elections quiz


To fetch the data, simply run
```
python fetch_and_transform_data.py
```
which should download 2 csv files to `data/results_2021.csv` and `data/questions_2021.csv`

To explore the notebook, set up an environment using conda/virtualenv and run
```
python -m pip install -r requirements.txt
python -m jupyter notebook
```

and simply navigate to the notebook and run it
