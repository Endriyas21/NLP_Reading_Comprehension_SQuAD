from transformers import BertTokenizerFast, AutoModelForQuestionAnswering, pipeline

def initialize_pipeline():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
    return qa_pipeline

def run_example(qa_pipeline, context, question):
    return qa_pipeline(question=question, context=context)

def generate_predictions(qa_pipeline, dev_data):
    predictions = []
    for article in dev_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question_id = qa['id']
                question = qa['question']

                try:
                    answer = qa_pipeline(question=question, context=context)
                    predictions.append({
                        'question_id': question_id,
                        'question': question,
                        'context': context,
                        'answer_text': answer['answer'],
                        'start': answer['start'],
                        'end': answer['end']
                    })
                except Exception as e:
                    print(f"Error processing question ID {question_id}: {e}")

    return predictions