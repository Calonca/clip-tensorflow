import numpy as np
from collections import Counter

# takes a list of concepts and returns a list of one-hot encoded labels
def one_hot_encode(concepts,labels,num_classes=1024):
    one_hot = np.zeros(num_classes)
    # one_hot[-1] = 1 # last element is the "other" class
    for concept in concepts:
        if concept in labels:
            one_hot[labels.index(concept)] = 1
        # else:
        #     one_hot[-1] = 1 # last element is the "other" class.
    return one_hot

def get_most_common_labels(df, num_classes=1024):
    # make a list of the most common concepts to use as labels
    concepts_l = df.concepts.to_list()
    concepts = [item for sublist in concepts_l for item in sublist]
    concepts_freq = Counter(concepts)
    concepts_freq = concepts_freq.most_common(num_classes)

    labels = [concept[0] for concept in concepts_freq]
    weights = [concept[1] for concept in concepts_freq]
    other_weight = len(concepts) - sum(weights)
    weights.append(other_weight)
    weights = max(weights)/np.array(weights)
    return labels,weights

#Prints lengt of dataframe and of duplicate captions in dataframe
def len_duplicate_caption(df):
    print(f'len df: {len(df)}')
    print(f'len df caption: {len(df.caption)}')
    print(f'len df caption unique: {len(df.caption.unique())}')

def print_metric(df,num_classes):
    # make a list of the most common concepts to use as labels
    concepts_l = df.concepts.to_list()
    concepts = [item for sublist in concepts_l for item in sublist]
    concepts_freq = Counter(concepts)
    concepts_lens = [len(concept) for concept in concepts]
    concepts_np = np.array(concepts)
    print(concepts_np.shape,concepts_np)
    print(f'total test samples: {len(df)}')
    print(f'total concepts with duplicates: {len(concepts)}')
    print(f'unique concepts: {len(concepts_freq)}')
    print(f'max concepts per sample: {max(concepts_lens)}')
    print(f'min concepts per sample: {min(concepts_lens)}')
    print(f'avg concepts per sample: {sum(concepts_lens)/len(concepts_lens)}')

    print()
    #print concepts of first 10 samples
    for i in range(5):
        print(f'concepts of sample {i}: {concepts_l[i]}')

    print(f'\nmost common concepts: ')
    print(concepts_freq.most_common(num_classes))

    total_concepts = concepts_freq.total()
    for concept,val in concepts_freq.most_common(num_classes):
        print(f'concept: {concept}, count: {val}, remaining: {total_concepts}')
        total_concepts -= val
    