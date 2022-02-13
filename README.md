## [“I’d rather just go to bed”: Understanding Indirect Answers](/https://aclanthology.org/2020.emnlp-main.601.pdf)

#### All experiments list (for relaxed labels) 
- [x] Baseline-MNLI
- [x] Baseline-BoolQ
- [x] Bert-circa-question-only 
- [x] Bert-circa-answer-only
- [x] Bert-circa
- [x] Bert-MNLI-circa
- [x] Bert-BoolQ-circa

#### Result Comparision

| Experiment              | F1 scores reported on the paper | Result in this code-base | Deviation(F1) |  Training Batch size as mentioned in the paper |
| ----------------------- | :-----------: | :--------:|  :--------:| :--------: |
| Baseline-MNLI           | 28.9 | 26.3 | :small_red_triangle_down: 2.6 | :x: |
| Baseline-BoolQ          | 62.7 | 65.0 | :small_red_triangle: 2.3 | :x: |
| Bert-circa-question-only| 56.0 | 55.4 | :small_red_triangle_down: 1.4 | :heavy_check_mark: |
| Bert-circa-answer-only  | 81.7 | 81.8 | :small_red_triangle: 0.1 | :heavy_check_mark: |
| Bert-circa              | 87.8 | 87.7 | :small_red_triangle_down: 0.1 | :heavy_check_mark: |
| Bert-MNLI-circa         | 88.2 | 88.2 | - | :x: |
| Bert-BoolQ-circa        | 87.1 | 87.2 | :small_red_triangle: 0.1 | :x: |


#### Hyper-parameters and 

| Experiment              | Learning Rate | Train Batch Size | Valid Batch Size |  Test Batch Size | Max Length | n_class | Dropout |
| ----------------------- | :-----------: | :--------:|  :--------:| :--------: |  :--------: | :--------: | :--------: |
| Baseline-MNLI           | 2e-5 | 26.3 | 8 | 4 | 4 | 512 | 3 | 0.3 |
| Baseline-BoolQ          | 2e-5 | 65.0 | 8 | 4 | 4 | 512 | 2 | 0.3 |
| Bert-circa-question-only| 3e-5 | 55.4 | 32 | 16 | 16 | 128 | 4 | 0.3 |
| Bert-circa-answer-only  | 2e-5 | 81.8 | 32 | 16 | 16 | 128 | 4 | 0.3 |
| Bert-circa              | 3e-5 | 87.7 | 32 | 16 | 16 | 128 | 4 | 0.3 |
| Bert-MNLI-circa         | 2e-5 | 88.2 | 8 | 4 | 4 | 512 | 4 | 0.3 |
| Bert-BoolQ-circa        | 3e-5 | 87.2 | 8 | 4 | 4 | 512 | 4 | 0.3 |
