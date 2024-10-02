import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_score(score):
    labels = ['Answer Score']
    scores = [score]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color='grey')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Confidence Score of the Answer')
    plt.show()

def plot_distributions(predictions):
    answer_lengths = [pred['end'] - pred['start'] for pred in predictions]
    start_positions = [pred['start'] for pred in predictions]

    plt.figure(figsize=(10, 6))
    plt.hist(answer_lengths, bins=30, edgecolor='black')
    plt.title('Distribution of Answer Lengths')
    plt.xlabel('Answer Length (characters)')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(start_positions, answer_lengths, alpha=0.5)
    plt.title('Answer Start Positions in Context')
    plt.xlabel('Start Position')
    plt.ylabel('Answer Length (characters)')
    plt.show()

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, size=20)
    plt.axis('off')
    plt.show()