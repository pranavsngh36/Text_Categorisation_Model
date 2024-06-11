#for text pre-processing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def graphs(x, y):
    df = pd.concat([x, y], axis=1)
    cats = sorted(['talk.politics.mideast',
                    'rec.autos',
                    'comp.sys.mac.hardware',
                    'alt.atheism',
                    'rec.sport.baseball',
                    'comp.os.ms-windows.misc',
                    'rec.sport.hockey',
                    'sci.crypt',
                    'sci.med',
                    'talk.politics.misc',
                    'rec.motorcycles',
                    'comp.windows.x',
                    'comp.graphics',
                    'comp.sys.ibm.pc.hardware',
                    'sci.electronics',
                    'talk.politics.guns',
                    'sci.space',
                    'soc.religion.christian',
                    'misc.forsale',
                    'talk.religion.misc'
                    ])
    data = {i : v for i,v in enumerate(cats)}
    df['category'] = [data[i] for i in df['label']]
    counts = df.category.value_counts().reset_index().sort_values(by='category')
    counts.columns = ['category', 'count']
    counts = counts.sort_values(by='category')
    counts['category'] = [str(i) + ': ' + v for i,v in enumerate(counts['category'])]
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    avg_counts = df.groupby('category')['word_count'].mean().round(0).reset_index()
    avg_counts = avg_counts.sort_values(by='category')

    plt.figure()
    sns.barplot(counts, x ='count',y = 'category', legend = False, palette = 'viridis').set_title('Documents distribution by Category')
    plt.ylabel('Category')
    plt.xlabel('Number of documents')

    plt.figure()
    # sns.barplot(avg_counts, x = 'word_count', y='category', hue = y, legend = False, palette = 'viridis').set_title('Average Word Count per Category')
    # plt.ylabel('Category')
    # plt.xlabel('Word Count')

    sns.boxplot(data=df, x='word_count', y='category', hue = y, legend = False, palette = 'viridis').set_title('Word Count Distribution per Category')
    plt.xscale('log')
    plt.ylabel('Category')
    plt.xlabel('Word Count')

def conf_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(15,10))
    sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')