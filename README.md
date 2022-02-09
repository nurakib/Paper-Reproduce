## [“I’d rather just go to bed”: Understanding Indirect Answers](/https://aclanthology.org/2020.emnlp-main.601.pdf)

#### All experiments list (for relaxed labels) 
- [ ] Baseline-MNLI
- [ ] Baseline-BoolQ
- [x] Bert-circa-question-only 
- [x] Bert-circa-answer-only
- [x] Bert-circa
- [ ] Bert-MNLI-circa
- [ ] Bert-BoolQ-circa

#### Result Comparision

| Experiment              | F1 scores reported on the paper | Result in this code-base | Deviation(%) |  Trained on Exact hyper-parameters mentioned in the paper |
| ----------------------- | :-----------: | :--------:|  :--------:|  
| Baseline-MNLI           | 28.9 |  |  | :x: |
| Baseline-BoolQ          | 62.7 |  |  | :x: |
| Bert-circa-question-only| 56.0 | 54.1 | :small_red_triangle_down: 3.39% | :heavy_check_mark: |
| Bert-circa-answer-only  | 81.7 | 81.1 | :small_red_triangle_down: 0.73% | :heavy_check_mark: |
| Bert-circa              | 87.8 | 87.2 | :small_red_triangle_down: 0.68%  | :heavy_check_mark: |
| Bert-MNLI-circa         | 88.2 |  |  | :x: |
| Bert-BoolQ-circa        | 87.1 |  |  | :x: |
