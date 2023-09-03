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

def to_lowercase(set_concepts):
    return {c.lower() for c in set_concepts}

def common_concepts_covering_all_dataset(df,num_classes, concepts_to_exclude = None):
    df1 = df.copy()
    df2 = df.copy()


    # df.concepts = df.concepts.apply(to_lowercase)
    # make a list of the most common concepts to use as labels
    concepts_l = df.concepts.to_list()
    concepts = [item.lower() for sublist in concepts_l for item in sublist]




    concepts_freq = Counter(concepts)

    if concepts_to_exclude:
        for concept_to_exclude in concepts_to_exclude:
            if concept_to_exclude in concepts_freq:
                del concepts_freq[concept_to_exclude]

    concepts_freq = concepts_freq.most_common(num_classes)

    #for ech concept add most common concept to list and remove from df until all row are covered
    concepts_covering_all_dataset = []
    df_len_before_removing = len(df)

    for concept in concepts_freq:

        df1 = df1[~df1['concepts'].apply(lambda x: concept[0] in x)]#remove rows containing concept

        concepts_covering_all_dataset.append((concept[0],concept[1],df_len_before_removing-len(df)))
        df_len_before_removing = len(df)
        if len(df) == 0:
            break

def get_captions_word_occurrences(df):

    captions_l = df.caption.to_list()
    captions_unique_string = ' '.join(captions_l)
    captions_words = [word.lower() for word in captions_unique_string.split(" ")]

    captions_freq = Counter(captions_words)

    return captions_freq



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
    