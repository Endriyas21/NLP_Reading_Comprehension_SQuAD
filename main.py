from install_dependencies import install_dependencies
from load_data import load_datasets, preprocess_squad
from qa_pipeline import initialize_pipeline, run_example, generate_predictions
from visualization import plot_score, plot_distributions, generate_wordcloud
from utils import check_gpu

# Install dependencies
install_dependencies()

# Check GPU availability
check_gpu()

# Load and preprocess datasets
train_data, dev_data = load_datasets()
train_contexts, train_questions, train_answers = preprocess_squad(train_data)
dev_contexts, dev_questions, dev_answers = preprocess_squad(dev_data)

# Initialize the question-answering pipeline
qa_pipeline = initialize_pipeline()

# Run an example
context = "My name is John Doe and I live in New York."
question = "What is my name?"
result = run_example(qa_pipeline, context, question)
print(result)

# Plot the score
plot_score(result['score'])

# Generate predictions
predictions = generate_predictions(qa_pipeline, dev_data)

# Save predictions to file
import json
with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=4)
print("Predictions saved to predictions.json")

# Load predictions from file
with open('predictions.json', 'r') as f:
    predictions = json.load(f)

# Plot distributions
plot_distributions(predictions)

# Generate word clouds
generate_wordcloud(" ".join(train_contexts[:2]), 'Sample Train Contexts')
generate_wordcloud(" ".join(train_questions[:2]), 'Sample Train Questions')
generate_wordcloud(" ".join([answer['text'] for answer in train_answers[:2]]), 'Sample Train Answers')