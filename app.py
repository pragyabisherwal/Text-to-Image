import streamlit as st
import numpy as np
import pickle
from Stage_1_GAN import *
from Stage_2_GAN import *

# Function to embed text into pickle file
def embed_text(text):
    # Replace this with your actual text embedding logic
    # For demonstration, using random embeddings for each text
    embedding = np.random.rand(1024)  # Assuming embedding size is 100
    return embedding

def embed_text_fun(texts):
    # File paths and directories
    embeddings_file_path_test = 'your_test_embeddings.pickle'
    class_info_file_path_test = 'your_class_info.pickle'
    filenames_file_path_test = 'your_filenames.pickle'
    cub_dataset_dir = 'your_cub_dataset_directory'

    # Embedding each text and creating embedded_data
    embedded_data = []
    for idx, text in enumerate(texts):
        embedding = embed_text(text)
        filename = f"image_{idx}.jpg"  # Replace with actual image filenames or identifiers
        embedded_data.append({
            'class_info': 'class_information_here',  # Replace with actual class information
            'filename': filename,  # Replace with actual filename
            'embedding': embedding
        })

    # Saving the embedded_data to a pickle file
    with open(embeddings_file_path_test, 'wb') as handle:
        pickle.dump(embedded_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_embedded_dataset(embeddings_file_path):
    with open(embeddings_file_path, 'rb') as handle:
        embedded_data = pickle.load(handle)
    
    # Extracting required information for generate_images_from_text function
    embeddings = [data['embedding'] for data in embedded_data]
    
    # Simulate other required data (for example purposes)
    z_dim = 100
    
    return embeddings, z_dim


# Function to generate images using Stage 1 and Stage 2 models
def generate_images_from_text(embeddings_file_path):
    # Load embedded dataset
    embeddings, z_dim = load_embedded_dataset(embeddings_file_path)
    # Ensure both arrays have the same number of samples
    num_samples = len(embeddings)
    
    # Initialize Stage 1 and Stage 2 models (replace with your actual initialization code)
    stage1_gen = build_stage1_generator()
    stage1_gen.load_weights("stage1_gen.h5")

    stage2_gen = build_stage2_generator()
    stage2_gen.load_weights("stage2_gen.h5")

    # Generate z_noise with the correct number of samples
    z_dim = 100  # Replace with the actual dimension
    z_noise = np.random.normal(0, 1, size=(num_samples, z_dim))

    # Convert embeddings and z_noise to NumPy arrays if not already
    embeddings = np.array(embeddings)
    z_noise = np.array(z_noise)
    # Generate images using loaded models
    stage1_generated_images, _ = stage1_gen.predict([embeddings, z_noise], verbose=3)
    stage2_generated_images, _ = stage2_gen.predict([embeddings, stage1_generated_images], verbose=3)
    
    return stage1_generated_images, stage2_generated_images

# Streamlit app with improved UI/UX and styling
def main():
    st.title('Text-to-Image Generation')
    st.markdown(
        "This app takes text as input, embeds it, and generates images using Stage 1 and Stage 2 models."
    )

    text_option = st.radio("Select text input option:", ('Enter text', 'Upload text file'))

    if text_option == 'Enter text':
        text_input = st.text_area("Enter text:")
        if st.button('Embed Text'):
            if text_input:
                texts = text_input.split('\n')
                embed_text_fun(texts)
                st.success('Text embedded successfully!')

    else:  # text_option == 'Upload text file'
        uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
        if uploaded_file is not None:
            file_text = uploaded_file.getvalue().decode("utf-8")
            st.write(file_text)
            if st.button('Embed Text from File'):
                texts = file_text.split('\n')
                embed_text_fun(texts)
                st.success('Text embedded successfully!')

    if st.button('Generate Images'):
        with st.spinner('Generating Images...'):
            # File path for embedded data
            embeddings_file_path = 'your_test_embeddings.pickle'

            # Generate images from text using the embedded dataset
            stage1_images, stage2_images = generate_images_from_text(embeddings_file_path)

        st.success('Images generated successfully!')
        # Scale the pixel values to [0.0, 1.0]
        stage1_images = np.clip(stage1_images, 0.0, 1.0)
        stage2_images = np.clip(stage2_images, 0.0, 1.0)

        col1, col2 = st.columns(2)
        with col1:
            # Display Stage 1 generated images
            st.subheader('Stage 1 Generated Image')
            for i, img in enumerate(stage1_images[:10]):
                st.image(img, caption=f'Stage 1 Image {i+1}', width=300)
        with col2:
            # Display Stage 2 generated images
            st.subheader('Stage 2 Generated Image')
            for i, img in enumerate(stage2_images[:10]):
                st.image(img, caption=f'Stage 2 Image {i+1}', width=300)

if __name__ == '__main__':
    main()
