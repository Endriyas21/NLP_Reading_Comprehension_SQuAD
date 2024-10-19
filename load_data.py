import json

def load_datasets():
    from google.colab import drive
    drive.mount('/content/drive/')

    train_data_path = '/content/drive/MyDrive/Datasets/SQuAD 2.0/train-v2.0.json'
    dev_data_path = '/content/drive/MyDrive/Datasets/SQuAD 2.0/dev-v2.0.json'

    try:
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(dev_data_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found.")
    except IOError as e:
        print(f"Error reading JSON file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
    else:
        print("JSON files loaded successfully.")
    
    return train_data, dev_data

def preprocess_squad(data):
    contexts = []
    questions = []
    answers = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if not qa['is_impossible']:
                    answer = qa['answers'][0]
                else:
                    answer = {'text': '', 'answer_start': 0, 'answer_end': 0}
                contexts.append(context)
                questions.append(question)
                answers.append(answer)
    return contexts, questions, answers