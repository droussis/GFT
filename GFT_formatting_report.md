
One parquet file per split. This is the exact schema of the official OmniAI-ZJU/NuminaMath-Cot-Distillation-100K (101,625 rows, verified from the file's parquet footer):

┌──────────────┬────────────────────────────────────────┬─────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│    column    │                  type                  │    required     │                                                  purpose                                                   │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ prompt       │ list<struct<role, content>>            │ yes             │ the chat messages; data.prompt_key=prompt. The chat template is applied by the dataset, so store the raw   │
│              │                                        │                 │ user turn including any instruction suffix                                                                 │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ reward_model │ struct<style, ground_truth>            │ yes             │ ground_truth is read by the reward manager (verl/workers/reward_manager/naive.py:82); for math it is the   │
│              │                                        │                 │ bare boxed answer, e.g. \frac{13}{6}                                                                       │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ extra_info   │ struct<answer, index, model_answer,    │ yes             │ see below; build_dataprop_from_gen_batch raises if extra_info is absent                                    │
│              │ question, split>                       │                 │                                                                                                            │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ source       │ string                                 │ yes with this   │ the example passes data.reward_fn_key="source", so a column with that name must exist (default would be    │
│              │                                        │ script          │ data_source)                                                                                               │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ data_source  │ string                                 │ keep it         │ standard verl column, also used in warning paths                                                           │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ ability      │ string                                 │ optional        │ unused by this path                                                                                        │
├──────────────┼────────────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ messages     │ list<struct<role, content>>            │ optional        │ unused by this path; kept in the released dataset for convenience                                          │
└──────────────┴────────────────────────────────────────┴─────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Constraints worth respecting

- Uniform struct. The file is loaded with datasets.load_dataset("parquet", ...), so every row must carry the same extra_info keys. A row with a null answer or a missing/short model_answer produces fewer candidates than teacher_count, while batch.repeat(teacher_count) (ray_trainer.py:1570) assumes a uniform count, and the union will fail on a size mismatch. Filter such rows out before writing.
- Teacher answers are tokenized with the training model's tokenizer, so responses longer than data.max_response_length are silently cut mid-solution. Filter or raise the limit rather than relying on truncation.
- Prompt length: filter_overlong_prompts=True drops overlong prompts before truncation='error' can raise, so those rows just disappear from the epoch.
- Validation file: needs only prompt, reward_model.ground_truth, source/data_source. model_answer is unused there (the teacher path only runs in fit()), and it is fine to keep the same schema.
- The released dataset's data_source value is a stale absolute path from the authors' machine (/opt/data/private/gwj/...). It is harmless because NuminaMath.compute_score ignores data_source, but set something sane in your own data.

Minimal converter

import pandas as pd

rows = []
for i, ex in enumerate(my_examples):
    question = ex["problem"] + " Let's think step by step and output the final answer within \\boxed{}."
    rows.append({
        "source": ex["source"],                    # reward_fn_key="source"
        "data_source": "my_dataset",
        "ability": "math",
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"style": "rule", "ground_truth": ex["boxed_answer"]},
        "extra_info": {
            "split": "train",
            "index": i,
            "question": ex["problem"],
            "answer": ex["expert_cot"],            # candidate 0 of the group
            "model_answer": ex["teacher_cots"],    # >= teacher_count - 1 entries, same length for every row
        },
    })

pd.DataFrame(rows).to_parquet("data/train.parquet")

Then point train_path at it in examples/gft_trainer/run_qwen2.5-math-1.5B.sh and fix the hardcoded custom_reward_function.path in 
