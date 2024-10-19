from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
import torch
from datasets import Dataset

def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def train_model(train_contexts, train_questions, train_answers):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    train_data = {
        'context': train_contexts,
        'question': train_questions,
        'answers': train_answers
    }
    train_dataset = Dataset.from_dict(train_data)
    tokenized_train_dataset = train_dataset.map(lambda x: prepare_train_features(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
    )

    trainer.train()
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')
    print("Model training complete and saved to './trained_model'")