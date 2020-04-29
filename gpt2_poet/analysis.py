import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if (len(word) == 0): # empty line
      return 0
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

def get_syllables_list_csv(file_path):
    haiku_syllables = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lines = row[0].strip().split('\n')
            if (len(lines) == 3): #strangely, there are some poems with 1 or 2 lines
                haiku_syllables.append(list(map(lambda x:syllables(x), lines)))
    #print(haiku_syllables)
    return haiku_syllables

def make_syllable_graphs(syllable_list, graphs_folder='graphs', graph_name='input', poem_type='Haiku'):
    #syllable_list is a 2-D list represented by haiku_syllables[haiku_num][num_syllables_in_line_n]

    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)

    output_path = os.path.join(graphs_folder, graph_name)

    poem_length = len(syllable_list[0])
    line_syllables = [np.zeros(15) for _ in range(0,poem_length)] #graph bar arrays

    for haiku in syllable_list:
        for i in range (0,poem_length):
            line_syllables[i][haiku[i]] += 1

    fig, axs = plt.subplots(1, poem_length, figsize=(15,4))
    fig.suptitle(f"Number of {poem_type} Occurrences by Syllable Count per Line")

    for i in range(0,poem_length):
        axs[i].bar(list(range(1,11)),line_syllables[i][:10])
        axs[i].set_title(f'Line {i+1}')
        axs[i].set_ylabel(f'Number of {poem_type} Occurrences')
        axs[i].set_xlabel('Number of Syllables')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(output_path)

    transposed = np.transpose(syllable_list)
    for i in range (0,poem_length):
        print(f"Mean syllables in line {i+1}: {np.mean(transposed[i])}")

    plt.show()

def analyze_greedy(num_poems=5):
    syllables = []
    for i in range (1,num_poems):
        syllables.append(get_syllables_list_csv(f'outputs/generated_{i}.csv')[0])

    # print(np.transpose(syllables))

    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    fig.suptitle(f"Number of Syllables by Epoch, per Line", fontsize='xx-large')
    for i in range(0,3):
        if (i == 0):
            input_mean = np.zeros(num_poems-1)
            input_mean += 3.64
            input_mode = np.zeros(num_poems-1)
            input_mode += 4
        elif (i == 1):
            input_mean = np.zeros(num_poems-1)
            input_mean += 5.12
            input_mode = np.zeros(num_poems-1)
            input_mode += 6
        elif (i == 2):
            input_mean = np.zeros(num_poems-1)
            input_mean += 4.15
            input_mode = np.zeros(num_poems-1)
            input_mode += 5
        axs[i].plot(list(range(1,num_poems)),input_mean)
        axs[i].plot(list(range(1,num_poems)),input_mode)
        axs[i].plot(list(range(1,num_poems)),np.transpose(syllables)[i])
        axs[i].set_title(f'Line {i+1}', fontsize='x-large')
        axs[i].set_ylabel(f'Number of Syllables', fontsize='x-large')
        axs[i].set_xlabel('Epoch', fontsize='x-large')
        axs[i].legend(['Input mean', 'Input mode', 'Greedy syllable count'])
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(top=0.85)
    plt.show()

def analyze_linegraph(num_epochs=5, poems_folder="outputs", line_title="mean syllables"):
    biglist = []
    for i in range (1,num_epochs):
      transposed = np.transpose(get_syllables_list_csv(f'{poems_folder}/generated_{i}.csv'))
      smalllist = []
      for j in range(0,3):
        smalllist.append(np.mean(transposed[j]))
      biglist.append(smalllist)
    # print(biglist)

    syllables = biglist
    # print(np.transpose(syllables))

    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    # fig, axs = plt.subplots(3, figsize=(5,15))
    fig.suptitle(f"Number of Syllables by Epoch, per Line", fontsize='xx-large')
    for i in range(0,3):
        if (i == 0):
          input_mean = np.zeros(num_epochs-1)
          input_mean += 3.64
          input_mode = np.zeros(num_epochs-1)
          input_mode += 4
        elif (i == 1):
          input_mean = np.zeros(num_epochs-1)
          input_mean += 5.12
          input_mode = np.zeros(num_epochs-1)
          input_mode += 6
        elif (i == 2):
          input_mean = np.zeros(num_epochs-1)
          input_mean += 4.15
          input_mode = np.zeros(num_epochs-1)
          input_mode += 5
        axs[i].plot(list(range(1,num_epochs)),input_mean)
        axs[i].plot(list(range(1,num_epochs)),input_mode)
        axs[i].plot(list(range(1,num_epochs)),np.transpose(syllables)[i])
        axs[i].set_title(f'Line {i+1}', fontsize='x-large')
        axs[i].set_ylabel(f'Number of Syllables', fontsize='x-large')
        axs[i].set_xlabel('Epoch', fontsize='x-large')
        axs[i].legend(['Input mean', 'Input mode', line_title])
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(top=0.85)
    # plt.subplots_adjust(wspace=0.3)
    # plt.subplots_adjust(hspace=0.3)
    # plt.subplots_adjust(top=0.94)
    plt.show()
