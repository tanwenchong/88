# ENsiRNA
Here are the detail training and testing script on Linux.

## Data Processing

The format of `your_file.csv` should follow the structure of the provided dataset. 

- **For ENsiRNA**:
  (Users should format their data into `your_file.csv` as per the [example](https://github.com/tanwenchong/ENsiRNA/blob/main/ENsiRNA/dataset/test.csv).)
  ```bash
  python -m data.get_pdb -f your_file.csv -p pdb_path # ENsiRNA
  ```

## Training

Before training, ensure you’ve updated the dataset directory in `config.json` according to the output from the data processing step.

```bash
GPU=0 bash train.sh config.json
```

## Testing

Test the model using the output file (`your_file.json`) from the data processing step.

```bash
GPU=0 bash test.sh checkpoint.pkl your_file.json saving_path
```



