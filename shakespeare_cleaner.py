import sys
import os
import re

def clean_text_folder(folder, new_folder):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    text_files = os.listdir(folder)
    for text_file in text_files:
        old_text_file = os.path.join(folder, text_file)
        new_text_file = os.path.join(new_folder, text_file)
        cleaned_text = clean_text(old_text_file)
        with open(new_text_file, 'w') as f:
            f.write(cleaned_text)


def clean_text(text_file):
    with open(text_file) as f:
        text = f.read()
        transformed_text = transform_text(text)
        cleaned_text = clean_spaces(transformed_text)
    return cleaned_text 


def transform_text(text):
    no_sections = re.sub(r'.*\n+=+', '', text)
    no_new_lines = re.sub(r'\n', ' ', no_sections)
    no_caps = re.sub(r'[A-Z]{2,}\,*', '', no_new_lines)
    no_double_dashes = re.sub(r'--', '.', no_caps)
    no_stage_directions = re.sub(r'\[.*?\]', '', no_double_dashes)
    no_octothorp = re.sub(r'#', '', no_stage_directions)
    return no_octothorp.lower()


def clean_spaces(text):

    def good_sentence(sentence):
        return len(sentence) > 0 and sentence.count('"') % 2 == 0

    sentences_w_spaces = text.split('.')
    sentences_wo_spaces = [sentence.strip() for sentence in sentences_w_spaces
                                            if good_sentence(sentence)]
    text = ('. ').join(sentences_wo_spaces)
    return text


if __name__ == '__main__':
    folder, new_folder = sys.argv[1], sys.argv[2]
    clean_text_folder(folder, new_folder)
