import numpy as np
import spacy
import time
from gensim.models import Word2Vec
import streamlit as st
import logging
from tensorflow.keras.models import load_model
from database import insert_user_details, close_connection, get_user_details

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


st.markdown("<h1 style='text-align: center; color: #4B0082; font-family: cursive;'>Consumer Complaint Disputer 🗳️</h1>", unsafe_allow_html=True)
project_details = """
The purpose of this project is to design and build a scalable machine learning pipeline
to predict given consumer complaint will be disputed or not.
## Team Members
- [Arya Chakraborty](https://www.linkedin.com/in/aryachakraborty/)
- [Rituparno Das](https://www.linkedin.com/in/rituparno-das/)
- [Prathamesh](https://www.linkedin.com/in/prathamesh-sawant/)
- [Bharadwaj](https://www.linkedin.com/in/bharadwaj-g-91b3a0178/)
"""
# Streamlit sidebar
st.sidebar.markdown("<h1 style='text-align: center; color:  #9b59b6 ;'>Will your Complaint get Disputed or not 🤔❓</h1>", unsafe_allow_html=True)
st.sidebar.markdown(project_details)

# Load models
def load_models():
    logging.info("Loading models...")
    lstm_model = load_model('lstm_model.h5')
    word2vec_model = Word2Vec.load("word2vec_model.bin")
    logging.info("Models loaded successfully.")
    return lstm_model, word2vec_model

# Get user inputs and concatenate them
def get_user_inputs():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        product_name = st.text_area('Product Name')
    with col2:
        issue = st.text_area('Issue(In detail)')
    with col3:
        company_public_response = st.text_area('Company Public Response')
    with col4:
        company_name = st.text_area('Company Name')

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        tags = st.text_area('Tags')
    with col6:
        submission_location = st.text_area('Submitted Through')
    with col7:
        company_response = st.text_area("Company's Response")
    with col8:
        timely_response = st.text_area('Timely Response(Yes/No)')

    user_inputs = [product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response]
    example_text = ', '.join(user_inputs)
    return example_text


def process_example_text(example_text, word2vec_model):

    nlp = spacy.blank("en")
    doc = nlp(example_text)
    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    example_vectors = [word2vec_model.wv[token] for token in cleaned_tokens if token in word2vec_model.wv]
    if not example_vectors:
        example_vectors.append(np.zeros(word2vec_model.vector_size))

    example_vector = np.mean(example_vectors, axis=0)
    return example_vector

# Make final prediction
def make_prediction(lstm_model, example_vector):
    final_vec = example_vector.reshape((1, example_vector.shape[0], 1))
    prediction = lstm_model.predict(final_vec)[0][0]
    return "There's a good probability your issue will be disputed." if prediction > 0.5 else "It's quite likely that your issue won't be contested."


# Main code
def main():
    logging.info("Starting the application...")
    lstm_model, word2vec_model = load_models()
    st.warning(
            """
            We appreciate you utilizing our AI-driven customer dispute system.
            Please be aware that the information you are giving is genuine and will be utilized to further refine the model.

            """
        )
    show_details = st.checkbox("Show Details")
    if show_details:
        example_text = get_user_inputs()

        if st.button('Show Prediction'):
            if all(input_val == '' for input_val in example_text.split(',')):
                st.warning("Give input first")
            else:
                example_vector = process_example_text(example_text, word2vec_model)
                prediction = make_prediction(lstm_model, example_vector)
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    #st.success('Done!')
                st.markdown(f"<h2 style='text-align: center; color: #4B0082;'>Prediction: {prediction}</h2>", unsafe_allow_html=True)
                logging.info(f"Prediction made: {prediction}")

                logging.info(f"Saving User Details")
if __name__ == '__main__':
    main()
