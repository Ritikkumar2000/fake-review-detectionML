import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Page config
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="wide")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        stopwords.words('english')
        return True
    except Exception as e:
        st.error(f"NLTK download failed: {e}")
        return False

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Preprocessing - MUST MATCH TRAINING EXACTLY
def preprocess_text(text):
    """
    EXACT preprocessing from training phase
    """
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    
    stemmer = PorterStemmer()
    
    # Step 1: Convert to lowercase
    text = str(text).lower()
    
    # Step 2: Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 3: Tokenize
    tokens = text.split()
    
    # Step 4: Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Step 5: Join back
    cleaned = ' '.join(tokens)
    
    return cleaned

# UPDATED Prediction function with detailed debugging
def predict_review(text, model, vectorizer):
    """
    Predict if review is fake or genuine
    Returns: prediction, genuine_confidence, fake_confidence, probabilities
    """
    if not text or not text.strip():
        return None, 0, 0, None
    
    # Preprocess
    cleaned = preprocess_text(text)
    
    # Debug: show cleaned text
    if st.session_state.get('show_debug', False):
        st.write(f"**Original (first 100 chars):** {text[:100]}")
        st.write(f"**After preprocessing:** '{cleaned}'")
    
    if not cleaned:
        return None, 0, 0, None
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Debug: show vectorization info
    if st.session_state.get('show_debug', False):
        st.write(f"**Feature vector shape:** {vectorized.shape}")
        st.write(f"**Non-zero features:** {vectorized.nnz}")
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    # Calculate confidences
    genuine_conf = probability[0] * 100
    fake_conf = probability[1] * 100
    
    # Debug: show raw probabilities
    if st.session_state.get('show_debug', False):
        st.write(f"**Raw probabilities:** Genuine={probability[0]:.6f}, Fake={probability[1]:.6f}")
        st.write(f"**Predicted class:** {prediction} ({'Fake' if prediction == 1 else 'Genuine'})")
    
    return prediction, genuine_conf, fake_conf, probability

# Initialize
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False

# Download NLTK data
nltk_ready = download_nltk_data()
if not nltk_ready:
    st.warning("⚠️ NLTK data download issue. App may not work correctly.")

# Load models
model, vectorizer = load_models()
if model is None or vectorizer is None:
    st.error("❌ Failed to load models. Please check that best_model.pkl and vectorizer.pkl are in the same directory.")
    st.stop()

# UI
st.title("🔍 Fake Review Detection System")
st.markdown("### Detect Computer-Generated Fake Reviews using Machine Learning")
st.success("✅ Model loaded successfully! (SVM - 89.5% Accuracy)")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model:** SVM (Support Vector Machine)
    
    **Performance:**
    - Accuracy: 89.5%
    - F1-Score: 89.4%
    - Trained on: 40,000+ reviews
    """)
    
    st.markdown("---")
    st.header("📊 Model Comparison")
    st.markdown("""
    | Model | F1-Score |
    |-------|----------|
    | 🥇 SVM | 89.4% |
    | 🥈 Random Forest | 85.0% |
    | 🥉 Naive Bayes | 84.6% |
    | XGBoost | 84.4% |
    | KNN | 71.1% |
    """)
    
    st.markdown("---")
    st.checkbox("🐛 Show Debug Info", key='show_debug')

# Main input
st.markdown("---")

# Input area
review_text = st.text_area(
    "📝 Enter a product review to analyze:",
    height=150,
    placeholder="Example: I purchased this product last month. The quality is decent for the price. Battery life is around 8 hours..."
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    analyze_btn = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

with col2:
    clear_btn = st.button("🔄 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

# Analysis
if analyze_btn and review_text.strip():
    with st.spinner("Analyzing review..."):
        prediction, genuine_conf, fake_conf, probability = predict_review(review_text, model, vectorizer)
        
        if prediction is None:
            st.error("❌ Could not analyze review. Please enter more text.")
        else:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Create 3 columns for better layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if prediction == 1:
                    st.error("### 🚨 FAKE REVIEW DETECTED")
                    st.markdown("**Classification:** Computer Generated (CG)")
                    st.markdown("**Reason:** Review shows patterns typical of AI-generated fake reviews")
                else:
                    st.success("### ✅ GENUINE REVIEW")
                    st.markdown("**Classification:** Original Review (OR)")
                    st.markdown("**Reason:** Review shows patterns of authentic user experience")
            
            with col2:
                st.metric("Genuine", f"{genuine_conf:.2f}%")
                if genuine_conf >= 70:
                    st.success("🎯 High")
                elif genuine_conf >= 50:
                    st.warning("⚡ Medium")
                else:
                    st.info("💡 Low")
            
            with col3:
                st.metric("Fake", f"{fake_conf:.2f}%")
                if fake_conf >= 70:
                    st.error("🚨 High")
                elif fake_conf >= 50:
                    st.warning("⚠️ Medium")
                else:
                    st.success("✓ Low")
            
            # Show confidence level
            confidence = max(genuine_conf, fake_conf)
            st.markdown("---")
            col_conf1, col_conf2 = st.columns([1, 3])
            with col_conf1:
                st.metric("Model Confidence", f"{confidence:.2f}%")
            with col_conf2:
                if confidence >= 85:
                    st.success("🎯 Very confident in this prediction")
                elif confidence >= 70:
                    st.warning("⚡ Moderately confident in this prediction")
                else:
                    st.info("💡 Low confidence - consider reviewing manually")

# Understanding section
st.markdown("---")
st.subheader("💡 Try Sample Reviews")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🚨 Fake-like Reviews:**")
    fake_samples = [
        "Best product ever! Amazing! 5 stars! Must buy now!",
        "Love it love it love it! Perfect perfect perfect!",
        "Worst ever! Terrible! Don't buy! Total waste!"
    ]
    
    for idx, sample in enumerate(fake_samples, 1):
        if st.button(f"Test Fake #{idx}", key=f"fake_{idx}"):
            with st.spinner("Testing..."):
                pred, gen_conf, fk_conf, prob = predict_review(sample, model, vectorizer)
                st.write(f"**Review:** {sample}")
                st.write(f"**Result:** {'FAKE' if pred == 1 else 'GENUINE'}")
                st.write(f"**Genuine:** {gen_conf:.1f}% | **Fake:** {fk_conf:.1f}%")
                if pred == 1:
                    st.success(f"✅ Correctly detected as FAKE")
                else:
                    st.warning(f"⚠️ Misclassified as GENUINE")

with col2:
    st.markdown("**✅ Genuine-like Reviews:**")
    genuine_samples = [
        "I purchased this last month. Quality is decent for the price. Battery lasts about 8 hours. Would recommend for budget users.",
        "Used for 2 weeks. Works as described. Setup was easy. Only issue is it's slightly heavier than expected.",
        "Good product overall. Shipping was fast. Material feels durable. Wish it had more color options."
    ]
    
    for idx, sample in enumerate(genuine_samples, 1):
        if st.button(f"Test Genuine #{idx}", key=f"gen_{idx}"):
            with st.spinner("Testing..."):
                pred, gen_conf, fk_conf, prob = predict_review(sample, model, vectorizer)
                st.write(f"**Review:** {sample}")
                st.write(f"**Result:** {'FAKE' if pred == 1 else 'GENUINE'}")
                st.write(f"**Genuine:** {gen_conf:.1f}% | **Fake:** {fk_conf:.1f}%")
                if pred == 0:
                    st.success(f"✅ Correctly detected as GENUINE")
                else:
                    st.warning(f"⚠️ Misclassified as FAKE")

# Explanation
with st.expander("ℹ️ What makes a review look FAKE?"):
    st.markdown("""
    **The model detects FAKE reviews based on linguistic patterns learned from 40,000+ examples:**
    
    ### 🚨 Fake Review Indicators:
    - Very short length (< 20 words)
    - Extreme emotions ("Best ever!", "Worst thing!")
    - Generic statements without specifics
    - Repetitive words and phrases
    - No personal experience details
    - Unrealistic perfection or complete disaster
    - Excessive punctuation (!!!, ***)
    
    ### ✅ Genuine Review Indicators:
    - Moderate length (30-100+ words)
    - Balanced opinions (pros AND cons)
    - Specific details (dates, numbers, features)
    - Personal experience ("I bought...", "Been using...")
    - Natural language with normal grammar
    - Constructive criticism
    - Contextual information (when, where, how)
    
    ### ⚠️ Important Note:
    The model analyzes **writing patterns and linguistic features**, not just keywords.
    Simply writing "this is fake" won't trigger detection - the model needs actual review-like content.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | Machine Learning Model: SVM with TF-IDF Features</p>
    <p>⚠️ This is an educational project. Results should be verified manually for critical decisions.</p>
</div>
""", unsafe_allow_html=True)
