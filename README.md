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
| Baseline-BoolQ          | 62.7 | 68.9 | :small_red_triangle: 6.2 | :x: |
| Bert-circa-question-only| 56.0 | 55.4 | :small_red_triangle_down: 1.4 | :heavy_check_mark: |
| Bert-circa-answer-only  | 81.7 | 81.1 | :small_red_triangle_down: 0.6 | :heavy_check_mark: |
| Bert-circa              | 87.8 | 87.7 | :small_red_triangle_down: 0.1 | :heavy_check_mark: |
| Bert-MNLI-circa         | 88.2 | 88.2 | - | :x: |
| Bert-BoolQ-circa        | 87.1 | 87.2 | :small_red_triangle: 0.1 | :x: |
