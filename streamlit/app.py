import streamlit as st
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# Streamlit page configuration
st.set_page_config(page_title="Melanoma Classification", layout="wide")

# CSS for better styling
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #004466; text-align: center;}
    h3 {color: #004466;}
    .e1b2p2ww10 {color: #004466;}
    small {color: #004466;}
    .stButton > button {background-color: #004466; color: white; border-radius: 8px;}
    .stMarkdown p {color: #004466;}
    .e16k0npc1 .e1bju1570 {color: #004466;}
    .e16k0npc0 {color: #004466;}
    .e1eexb540 .e1nzilvr5 p {color: #004466;}
    .st-c7 {background-color: #7F8487;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Model file paths
MODEL_PATHS = {
    "VGG-19 Without CLAHE": "VGG-19/model_best_tanpa_clahe.onnx",
    "VGG-19 With CLAHE": "VGG-19/model_best_dengan_clahe.onnx",
    "ResNet-50 Without CLAHE": "ResNet-50/model_best_tanpa_clahe.onnx",
    "ResNet-50 With CLAHE": "ResNet-50/model_best_dengan_clahe.onnx",
}

# Mean and std for normalization
MEAN_STD = {
    "Without CLAHE": ([0.7206, 0.5360, 0.2959], [0.0448, 0.0475, 0.0406]),
    "With CLAHE": ([0.7288, 0.5902, 0.4875], [0.0497, 0.0495, 0.0502]),
}

# Cache the models
@st.cache_resource
def load_model(model_key):
    return ort.InferenceSession(MODEL_PATHS[model_key])

# Image preprocessing
def preprocess_image(image, mean, std):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(image).unsqueeze(0).numpy()

# Image classification
def classify_image(image, model_session, mean, std):
    image = preprocess_image(image, mean, std)
    inputs = {model_session.get_inputs()[0].name: image}
    start_time = time.time()
    outputs = model_session.run(None, inputs)
    inference_time = time.time() - start_time
    probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
    return probabilities[0, 0], probabilities[0, 1], inference_time

# Inisialisasi session_state untuk menyimpan hasil klasifikasi
if "results" not in st.session_state:
    st.session_state.results = {}

# Main layout
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.title("🩺 Melanoma Skin Cancer Classification")
    st.markdown("---")

    # Multi-select untuk memilih model
    selected_models = st.multiselect(
        "Choice Model Klasifikasi:", list(MODEL_PATHS.keys()), default=list(MODEL_PATHS.keys())[:2]
    )

    # Load models
    models = {key: load_model(key) for key in MODEL_PATHS.keys()}

    col_without, col_with = st.columns(2)

    inference_times = {key: [] for key in MODEL_PATHS.keys()}
    processing_done = {key: False for key in MEAN_STD.keys()}
    start_times = {}
    end_times = {}

    for model_key in ["Without CLAHE", "With CLAHE"]:
        with (col_without if "Without" in model_key else col_with):
            st.subheader(f"📷 {model_key}")
            uploaded_files = st.file_uploader(f"Upload 10 Images ({model_key})...", type=["jpg", "png"], accept_multiple_files=True, key=model_key)

            if uploaded_files:
                # Reset hasil jika upload baru
                current_key = f"{model_key}_num_files"
                if st.session_state.get(current_key, 0) != len(uploaded_files):
                    st.session_state.results = {}
                st.session_state[current_key] = len(uploaded_files)

                if len(uploaded_files) == 10:
                    images = [Image.open(file) for file in uploaded_files]
                    start_times[model_key] = time.time()

                    for i, img in enumerate(images):
                        st.image(img, caption=f"Image {i+1}", use_column_width=True)

                        for model_name in selected_models:
                            if model_name.endswith(model_key):
                                key = (model_name, model_key, i)
                                if key in st.session_state.results:
                                    melanoma_prob, normal_prob, inference_time = st.session_state.results[key]
                                else:
                                    melanoma_prob, normal_prob, inference_time = classify_image(img, models[model_name], *MEAN_STD[model_key])
                                    st.session_state.results[key] = (melanoma_prob, normal_prob, inference_time)

                                inference_times[model_name].append(inference_time)
                                st.subheader(f"{model_name}")
                                st.write(f"Melanoma: {melanoma_prob * 100:.2f}%")
                                st.write(f"Normal: {normal_prob * 100:.2f}%")
                                st.write(f"⏱️ Inference Time: {inference_time:.4f} sec")

                        # Perhitungan rata-rata
                        melanoma_avg_list = []
                        normal_avg_list = []

                        for m in selected_models:
                            if m.endswith(model_key):
                                key = (m, model_key, i)
                                if key in st.session_state.results:
                                    mel, norm, _ = st.session_state.results[key]
                                else:
                                    mel, norm, _ = classify_image(img, models[m], *MEAN_STD[model_key])
                                    st.session_state.results[key] = (mel, norm, _)
                                melanoma_avg_list.append(mel)
                                normal_avg_list.append(norm)

                        melanoma_avg = np.mean(melanoma_avg_list)
                        normal_avg = np.mean(normal_avg_list)

                        if melanoma_avg > normal_avg:
                            st.error("⚠️ Melanoma detected!")
                        else:
                            st.success("✅ Skin condition is normal.")

                        st.markdown("---")

                    end_times[model_key] = time.time()
                    processing_done[model_key] = True
                else:
                    st.warning("Please upload exactly 10 images.")

    # Display average inference time
    if any(inference_times.values()):
        st.subheader("📊 Average Inference Time:")
        for model_name, times in inference_times.items():
            if times:
                st.write(f"⏳ {model_name}: {np.mean(times):.4f} sec")

    st.markdown("---")
    st.caption("⚠️ This application is for informational purposes only and does not replace medical consultation.")
