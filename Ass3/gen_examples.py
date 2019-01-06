import random
import string

STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

EXAMPLES_NUM = 500
DATA_SIZE = 5000
POS_EXAMPLE = "pos_examples"
NEG_EXAMPLE = "neg_examples"
TRAIN = "data/train"
TEST = "data/dev"
POLY_REGEX = "regex_poly_examples"
WW_REGEX = "regex_ww_examples"


def generate_examples(examples_num, rand_flag, neg_flag, file_path):
    """
    generate positive/negative examples, depending on flag
    :param examples_number: examples number
    :param rand:
    :param flag: True: positive examples
                 False: negative examples
    :param file_path: path
    """
    with open(file_path, 'w') as file:
        for i in range(examples_num):
            # generate random examples
            if (rand_flag):
                tag = random.choice([True, False])
                single_example = generate_single_example(tag)
                file.write(single_example + '\t' + str(tag) + '\n')
            # generate discrete examples
            else:
                single_example = generate_single_example(neg_flag)
                file.write(str(single_example) + '\n')


def generate_single_example(neg_flag):
    """
    generate single example by a given flag
    :param pos_flag: neg_flag is a boolean.
                     True: negative   False: positive
    :return: example representation
    """
    example = ""
    example += digits_sequence()
    example += character_sequence('a')
    example += digits_sequence()
    # negative  -> 'c'  before 'b'
    if neg_flag:
        example += character_sequence('c')
        example += digits_sequence()
        example += character_sequence('b')
    # positive -> 'b' before 'c'
    else:
        example += character_sequence('b')
        example += digits_sequence()
        example += character_sequence('c')
    example += digits_sequence()
    example += character_sequence('d')
    example += digits_sequence()
    return example


def character_sequence(character):
    """
    create characters sequence by a given character.
    stop creating sequence by random chance of 10%.
    :return: sequence
    """
    squence = ""
    while True:
        squence += character
        # chance of 10% to stop creating sequence
        if random.randint(1, 10) == 1:
            break
    return squence


def digits_sequence():
    """
    create random sequence digits of digits between 1-9.
    stop creating sequence by random chance of 10%.
    :return: sequence
    """
    squence = ""
    while True:
        rnd_digit = random.randint(1, 9)
        squence += str(rnd_digit)
        # chance of 10% to stop creating sequence
        if random.randint(1, 10) == 1:
            break
    return squence


def generate_palindrome(examples_size, output_fname):
    """
    generate palindrome sequence
    :param examples_size: size of data
    """
    words = []
    chars_and_digits = string.ascii_lowercase + string.digits
    for i in range(examples_size):
        sequence = "".join([random.choice(chars_and_digits) for _ in range(random.randint(1, 50))])
        reversed = sequence[::-1]
        word = sequence + '#' + reversed
        words.append(word)
    with open(output_fname, 'w+') as file:
        for word in words:
            file.write(word + '\n')


def generate_ww_sequence(examples_size, output_fname):
    """
    generate ww sequence
    :param examples_size:
    :return:
    """
    words = []
    chars_and_digits = string.ascii_lowercase + string.digits
    for i in range(examples_size):
        # gets new number for positive seq.
        sequence = "".join([random.choice(chars_and_digits) for _ in range(random.randint(1, 50))])
        reversed = sequence
        word = sequence + reversed
        words.append(word)
    with open(output_fname, 'w+') as file:
        for word in words:
            file.write(word + '\n')


if __name__ == "__main__":

    # pos and neg examples
    generate_examples(EXAMPLES_NUM, False, True, NEG_EXAMPLE)
    generate_examples(EXAMPLES_NUM, False, False, POS_EXAMPLE)

    # train and dev data sets of examples
    generate_examples(round(DATA_SIZE * 9/10), True, None, TRAIN)
    generate_examples(round(DATA_SIZE * 1/10), True, None, TEST)

    ##         --following is answering part 2--             ##
    ##   two examples of regex that RNN cannot distinguish   ##

    # palindrome regex
    generate_palindrome(EXAMPLES_NUM, POLY_REGEX)
    # double word regex
    generate_ww_sequence(EXAMPLES_NUM, WW_REGEX)

