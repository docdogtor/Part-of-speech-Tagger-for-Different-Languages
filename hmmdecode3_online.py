import sys

test_path = sys.argv[1]


def readModel():
    model = "hmmmodel.txt"
    model_file = open(model, "r")
    emission_count = dict()
    emission_tag_sum = dict()
    transition_count = dict()
    transition_tag_sum = dict()

    first_line = model_file.readline()
    emission_size = int(first_line)
    for i in range(emission_size):
        model_line = model_file.readline()
        line = model_line.split()
        if line[0] not in emission_count:
            emission_count[line[0]] = {line[1]: int(line[2])}
            emission_tag_sum[line[0]] = int(line[3])
        else:
            emission_count[line[0]][line[1]] = int(line[2])

    first_line = model_file.readline().split()
    pie_start = first_line[0]
    second_line = model_file.readline().split()
    pie_end = second_line[0]

    model_lines = model_file.readlines()
    for model_line in model_lines:
        line = model_line.split()
        if line[0] not in transition_count:
            transition_count[line[0]] = {line[1]: int(line[2])}
            transition_tag_sum[line[0]] = int(line[3])
        else:
            transition_count[line[0]][line[1]] = int(line[2])
    model_file.close()
    return emission_count, emission_tag_sum, transition_count, transition_tag_sum, pie_start, pie_end


emission_count, emission_sum, transition_count, transition_sum, pie_start, pie_end = readModel()

state_num = len(emission_sum)
sum_all = 0
vocabulary = set()
emission_model = dict()
emission_modify = dict()
transition_model = dict()
tag_distribution = dict()

bound = state_num // 7

state = []
for tag in emission_count:
    emission_modify[tag] = len(emission_count[tag])
    sum_all += emission_modify[tag]
    state.append(tag)
    for word in emission_count[tag]:
        vocabulary.add(word)

value_list = sorted(emission_modify.values())

for tag in state:
    if emission_modify[tag] < value_list[-1*bound]:
        value = 0
    else:
        value = 1
    tag_distribution[tag] = value

# emission model
for tag in state:
    for word in emission_count[tag]:
        if tag not in emission_model:
            emission_model[tag] = {word: emission_count[tag][word]/emission_sum[tag]}
        else:
            emission_model[tag][word] = emission_count[tag][word]/emission_sum[tag]

# transition model
transition_model[pie_start] = {state[0]: 0}
for tag in state:
    if tag not in transition_count[pie_start]:
        transition_count[pie_start][tag] = 0
    transition_model[pie_start][tag] = (transition_count[pie_start][tag] + 1)/(transition_sum[pie_start] + state_num)

state_end_num = state_num + 1
for previous_tag in state:
    transition_model[previous_tag] = {previous_tag: 0}
    if previous_tag not in transition_count:
        for tag in state:
            transition_model[previous_tag][tag] = 1/state_end_num
        transition_model[previous_tag][pie_end] = 1/state_end_num
    else:
        for tag in state:
            if tag not in transition_count[previous_tag]:
                transition_count[previous_tag][tag] = 0
            transition_model[previous_tag][tag] = (transition_count[previous_tag][tag] + 1)/(transition_sum[previous_tag] + state_end_num)
        if pie_end not in transition_count[previous_tag]:
            transition_count[previous_tag][pie_end] = 0
        transition_model[previous_tag][pie_end] = (transition_count[previous_tag][pie_end] + 1) / (transition_sum[previous_tag] + state_end_num)

file_read = open(test_path, 'r')
file_write = open("hmmoutput.txt", 'w')
for line in file_read:
    sentence = line.split()
    step_num = len(sentence)
    viterbi = [[0 for y in range(state_num)] for x in range(step_num + 1)]
    backpointer = [[0 for y in range(state_num)] for x in range(step_num + 1)]
    for x in range(step_num):
        flag = 0
        word = sentence[x].lower()
        if word not in vocabulary:
            flag = 1

        if x == 0:
            start_tag = pie_start
            previous_num = []
            for y in range(state_num):
                tag = state[y]
                if word not in emission_model[tag]:
                    if flag == 0:
                        emission_model[tag][word] = 0
                    else:
                        emission_model[tag][word] = tag_distribution[tag]

                viterbi[x][y] = transition_model[start_tag][tag]*emission_model[tag][word]
                backpointer[x][y] = start_tag
                if viterbi[x][y] != 0:
                    previous_num.append(y)
        else:
            temporary_num = []
            for y in range(state_num):
                if len(previous_num) == 0:
                    print("BREAK")
                max_probability = 0
                max_pointer = state[previous_num[0]]
                tag = state[y]
                for p_num in previous_num:
                    previous_tag = state[p_num]
                    if word not in emission_model[tag]:
                        if flag == 0:
                            emission_model[tag][word] = 0
                        else:
                            emission_model[tag][word] = tag_distribution[tag]
                    current_probability = viterbi[x-1][p_num]*transition_model[previous_tag][tag]*emission_model[tag][word]
                    if max_probability < current_probability:
                        max_probability = current_probability
                        max_pointer = p_num
                viterbi[x][y] = max_probability
                backpointer[x][y] = max_pointer
                if viterbi[x][y] != 0:
                    temporary_num.append(y)
            previous_num = temporary_num

    x = step_num
    y = 0
    max_probability = 0
    max_pointer = state[previous_num[0]]
    tag = pie_end

    for p_num in previous_num:
        previous_tag = state[p_num]
        current_probability = viterbi[x - 1][p_num] * transition_model[previous_tag][tag]
        if max_probability < current_probability:
            max_probability = current_probability
            max_pointer = p_num
    viterbi[x][y] = max_probability
    backpointer[x][y] = max_pointer

    best_path = dict()
    best_path[x-1] = backpointer[x][y]

    for x in reversed(range(step_num - 1)):
        y = backpointer[x+1][best_path[x+1]]
        best_path[x] = y

    for x in range(step_num):
        word = sentence[x]
        y = int(best_path[x])
        file_write.write("%s" % word)
        file_write.write("/")
        file_write.write("%s " % state[y])
    file_write.write("\n")

