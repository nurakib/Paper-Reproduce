# Circa

### Running the Code

#### Arguments:
```
  p.add_argument('--experiment_name', type=str, help='Name of the experiment.', default=None)
  p.add_argument('--train_data', type=str, help='Path to the train data.', default=None)
  p.add_argument('--valid_data', type=str, help='Path to the test data.', default=None)
  p.add_argument('--test_data', type=str, help='Path to the validation data.', default=None)
  p.add_argument('--sample_percentage', type=int, help='Percentage of instances to consider from the datasets', default=100)
  p.add_argument('--train_batch_size', type=int, help='Training dataset batch size', default=32)
  p.add_argument('--valid_batch_size', type=int, help='Validation dataset batch size', default=16)
  p.add_argument('--test_batch_size', type=int, help='Test dataset batch size', default=16)

  p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='bert-base-uncased')
  p.add_argument('--trained_model', type=str, help='Model path.', default=None)
  p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=128)
  p.add_argument('--n_class', type=int, help='Number of output class', default=4)
  p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=3)
  p.add_argument('--lr', type=float, help='Learning rate', default=3e-5)
  p.add_argument('--dropout', type=float, help='Dropout rate', default=0.3)

  p.add_argument('--save_model_dir', type=str, help='Save Model directory after training.', default='.')
  p.add_argument('--result_dir', type=str, help='Save Model directory after training.', default='.')
``` 

#### Running 

###### Train a bert-base-uncased base model
```
python train.py --train_data ../datasets/boolq/boolq_train.csv --valid_data ../datasets/boolq/boolq_valid.csv \
                --sample_percentage 5 --train_batch_size 16 --valid_batch_size 8 --encoder_model bert-base-uncased \
                --max_length 512 --epochs 3 --lr 3e-5 --save_model_dir ../saved_models/ --experiment_name boolq_base
```

###### Finetuning on a the trained model
```
python train.py --train_data ../datasets/circa/circa_train.csv --valid_data ../datasets/circa/circa_valid.csv \
                --sample_percentage 5 --train_batch_size 16 --valid_batch_size 8 --encoder_model bert-base-uncased \
                --trained_model ../saved_models/model_boolq_base.bin --max_length 512 --epochs 3 --lr 3e-5 \
                --save_model_dir ../saved_models/ --experiment_name boolq_base_circa
```

###### Testing from a pretrained model

```
python test.py --test_data ../datasets/circa/circa_test.csv --trained_model ../saved_models/model_boolq_base_circa.bin \
               --result_dir ../results/ --sample_percentage 5 --experiment_name boolq_base_circa

```

### Setting up the code environment

```
$ pip install -r requirements.txt
```
