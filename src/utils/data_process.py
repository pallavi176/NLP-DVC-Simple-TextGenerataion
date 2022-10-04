import re
import string
import pandas as pd

def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_word(txt, word='other'):
    return re.sub(word, '', txt)

def preprocess_df(input_data, sheet_name):
    df = pd.read_excel(input_data, sheet_name = sheet_name)

    # Read description column
    codes_desc = df[['Descriptions']]

    # Remove punctuation
    codes_desc['Descriptions'] = codes_desc['Descriptions'].apply(lambda x: remove_punc(x))

    #Remove All Capital Sentences
    codes_desc = codes_desc[~codes_desc.Descriptions.map(lambda x: x.isupper())]

    # Lower all caps
    codes_desc['Descriptions'] = codes_desc['Descriptions'].apply(lambda x: x.lower())

    #Remove Other keyword
    codes_desc = codes_desc[~codes_desc.Descriptions.map(lambda x: x.lower() == 'other')]

    codes_desc['Descriptions'] = codes_desc['Descriptions'].apply(lambda x: remove_word(x))

    text = " ".join(codes_desc['Descriptions'])

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text