import argparse
import datetime

def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
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

    return p.parse_args()
