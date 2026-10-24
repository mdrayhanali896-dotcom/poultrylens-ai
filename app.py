import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="PoultryLens AI", page_icon="🐔", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#07120f;--panel:#0b211a;--lime:#c8f169;--mint:#b9e8cb;--text:#ffffff;--muted:#d0ddd5}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;font-size:17px}
.stApp{background:radial-gradient(ellipse at 80% 0%,#102c20 0%,#07120f 48%);color:#ffffff}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1400px}
[data-testid="stSidebar"]{background:#050e0b;border-right:1px solid #294a38}
[data-testid="stSidebar"] *{color:#f7fff9!important;font-size:16px}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.02em;color:#ffffff!important}
h1{font-size:2.25rem!important}
h2{font-size:1.8rem!important}
h3{font-size:1.35rem!important}
p,li,label,.stMarkdown,.stText,div[data-testid="stCaptionContainer"]{color:#f0f7f2}
p,li{font-size:17px;line-height:1.65}
input,textarea,[data-baseweb="select"] *{color:#ffffff!important}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#102b20,#091a14);border:1px solid #31543f;padding:18px 20px;border-radius:18px}
div[data-testid="stMetricLabel"]{color:#e0ece4!important;font-size:15px}
div[data-testid="stMetricValue"]{color:#d5ff7d!important;font-weight:800}
.stButton>button,.stDownloadButton>button{background:#c8f169;color:#0b2117;border:0;border-radius:12px;font-weight:800;padding:.7rem 1rem;font-size:16px}
.stButton>button:hover,.stDownloadButton>button:hover{background:#d8ff8c;color:#0b2117;border:0}
div[data-testid="stFileUploader"]{background:#0a2118;border:1px dashed #8bbd91;border-radius:16px;padding:12px}
div[data-testid="stFileUploader"] *{color:#ffffff!important}
hr{border-color:#31513e}
.pill{display:inline-block;border:1px solid #638452;background:#193526;color:#efffd5;padding:6px 11px;border-radius:999px;font-size:14px;font-weight:700;letter-spacing:.04em}
.hero{background:linear-gradient(120deg,rgba(20,54,36,.98),rgba(7,25,18,.98));border:1px solid #3b6848;padding:32px;border-radius:24px;margin-bottom:20px}
.hero h1{font-size:clamp(2.2rem,4vw,3.3rem)!important;margin:12px 0 8px;color:#ffffff!important}
.hero p{max-width:760px;font-size:1.1rem;line-height:1.75;color:#e0eee5!important}
.small{font-size:1rem;color:#d0dfd5!important}
.panel{background:linear-gradient(145deg,#0c241a,#081a13);border:1px solid #31543e;border-radius:18px;padding:20px;height:100%}
.panel h3{color:#ffffff!important;margin-top:0}
.panel p{color:#e1eee5!important}
.kicker{color:#d5ff7d;text-transform:uppercase;font-size:.9rem;letter-spacing:.1em;font-weight:800}
.stSelectbox label,.stRadio label,.stCheckbox label,.stSlider label{color:#ffffff!important;font-size:16px!important}
[data-testid="stDataFrame"] *{color:#ffffff!important}
</style>
""", unsafe_allow_html=True)

MODEL_RESULTS={
"Hybrid Model":{"accuracy":98.45,"f1":98.45,"size":"33.02 MB","note":"EfficientNetB0 + MobileNetV2"},
"EfficientNetB0":{"accuracy":98.14,"f1":98.15,"size":"Not recorded","note":"ImageNet transfer learning"},
"MobileNetV2":{"accuracy":97.10,"f1":97.12,"size":"12.99 MB","note":"Lightweight architecture"},
"InceptionV3":{"accuracy":95.24,"f1":95.26,"size":"90.31 MB","note":"Multi-scale feature extraction"},
"ResNet50":{"accuracy":89.44,"f1":89.50,"size":"Not recorded","note":"Residual learning"}}
CLASSES=["Cocci","Healthy","Salmo"]

@st.cache_resource(show_spinner=False)
def load_hybrid_model():
    from pathlib import Path
    path=Path(__file__).parent/"Hybrid_Model.keras"
    if not path.exists(): return None,f"Model file not found: {path.name}"
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path),None
    except Exception as e: return None,str(e)

def predict_image(image):
    model,error=load_hybrid_model()
    if model is None:
        st.error(f"Could not load Hybrid_Model.keras: {error}"); return
    image=image.convert("RGB").resize((224,224))
    arr=np.asarray(image,dtype=np.float32)
    try: scores=np.asarray(model.predict(np.expand_dims(arr,0),verbose=0))[0]
    except Exception as e: st.error(f"Prediction failed: {e}"); return
    idx=int(np.argmax(scores)); label=CLASSES[idx] if idx<len(CLASSES) else f"Class {idx}"
    st.success(f"Prediction: **{label}**")
    st.metric("Model confidence",f"{float(scores[idx]):.2%}")
    st.progress(float(np.clip(scores[idx],0,1)))
    st.write("Class scores")
    st.bar_chart({CLASSES[i] if i<len(CLASSES) else f"Class {i}":float(v) for i,v in enumerate(scores)})
    st.caption("Verify the output-class order and preprocessing against the training notebook before public release.")

with st.sidebar:
    st.markdown("<div style='font-size:2rem'>🐔</div><h2 style='margin:0'>PoultryLens <span style='color:#c8f169'>AI</span></h2><p class='small'>SMART POULTRY HEALTH</p>",unsafe_allow_html=True)
    st.divider()
    page=st.radio("WORKSPACE",["Overview","Live Prediction","Model Lab","Hybrid Architecture","Evaluation","Dataset & Methodology"])
    st.divider()
    st.markdown("<div class='panel'><div class='kicker'>SYSTEM STATUS</div><h3 style='margin:8px 0'>Ready to analyze</h3><p class='small'>Upload a chicken image to get a model prediction.</p></div>",unsafe_allow_html=True)
    st.caption("Research prototype · Not a veterinary diagnosis")

if page=="Overview":
    st.markdown("""<div class="hero"><span class="pill">AI-POWERED POULTRY HEALTH</span><h1>See the signs.<br><span style="color:#c8f169">Protect the flock.</span></h1><p>PoultryLens AI analyzes chicken images to classify them as Cocci, Healthy, or Salmo using a deep-learning hybrid model.</p></div>""",unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Hybrid accuracy","98.45%","Test-set result"); b.metric("Macro F1-score","98.45%","Balanced evaluation"); c.metric("Output classes","3","Cocci · Healthy · Salmo"); d.metric("Hybrid model size","33.02 MB","Saved model")
    st.markdown("## Your workspace")
    cols=st.columns(3)
    for col,emoji,title,desc in zip(cols,["📷","🧪","🧬"],["Live Prediction","Model Lab","Hybrid Architecture"],["Upload an image or use your camera.","Compare five model experiments.","Explore the feature-fusion pipeline."]):
        with col: st.markdown(f"<div class='panel'><div style='font-size:1.8rem'>{emoji}</div><h3>{title}</h3><p>{desc}</p></div>",unsafe_allow_html=True)
    st.info("Research prototype only. Have predictions reviewed by a poultry-health professional.")
elif page=="Live Prediction":
    st.markdown("<div class='kicker'>IMAGE ANALYSIS</div>",unsafe_allow_html=True); st.title("Live Disease Prediction")
    st.write("Upload a chicken image or capture a photo. The saved Hybrid Model will analyze it.")
    mode=st.radio("Image source",["Upload image","Use camera"],horizontal=True)
    image=None
    if mode=="Use camera":
        capture=st.camera_input("Take a photo")
        if capture: image=Image.open(capture)
    else:
        upload=st.file_uploader("Choose an image",type=["jpg","jpeg","png","webp"])
        if upload: image=Image.open(upload)
    if image:
        left,right=st.columns([1,1])
        with left: st.image(image,caption="Selected image",use_container_width=True)
        with right:
            st.markdown("<div class='panel'><div class='kicker'>MODEL</div><h3>Hybrid EfficientNetB0 + MobileNetV2</h3><p>Output classes: Cocci, Healthy, Salmo.</p></div>",unsafe_allow_html=True)
            if st.button("Analyze image",type="primary",use_container_width=True): predict_image(image)
    else: st.markdown("<div class='panel'><h3>How it works</h3><p>1. Select or capture a clear image.</p><p>2. Click Analyze image.</p><p>3. Review the predicted class and confidence.</p></div>",unsafe_allow_html=True)
elif page=="Model Lab":
    st.markdown("<div class='kicker'>EXPERIMENT COMPARISON</div>",unsafe_allow_html=True); st.title("Model Lab")
    st.write("Recorded test results for all five model experiments.")
    for name,r in MODEL_RESULTS.items():
        with st.container(border=True):
            left,mid,right=st.columns([2,1,1])
            left.markdown(f"**{name}**  \n<span class='small'>{r['note']}</span>",unsafe_allow_html=True)
            mid.metric("Accuracy",f"{r['accuracy']:.2f}%"); right.metric("Macro F1",f"{r['f1']:.2f}%")
            st.progress(r["accuracy"]/100); st.caption(f"Model size: {r['size']}")
    st.caption("Verify these previously recorded values against the final evaluation notebook before publication.")
elif page=="Hybrid Architecture":
    st.markdown("<div class='kicker'>MODEL DESIGN</div>",unsafe_allow_html=True); st.title("Hybrid Architecture")
    st.write("Two pretrained feature extractors feed a shared classifier.")
    for n,title,desc in [("01","Input image","RGB image resized to 224 × 224"),("02A","EfficientNetB0 branch","Pretrained feature extraction"),("02B","MobileNetV2 branch","Pretrained feature extraction"),("03","Feature fusion","Concatenate feature representations"),("04","Normalization","Batch Normalization"),("05","Classifier head","Dense 256 · ReLU · Dropout 0.4"),("06","Prediction","Dense 3 · Softmax")]:
        st.markdown(f"<div class='panel' style='margin-bottom:10px;display:flex;gap:18px;align-items:center'><div style='font-size:1.2rem;font-weight:800;color:#c8f169;min-width:48px'>{n}</div><div><h3 style='margin:0'>{title}</h3><p style='margin:4px 0 0'>{desc}</p></div></div>",unsafe_allow_html=True)
    st.warning("Confirm model layers and preprocessing in the saved .keras file before treating this as a formal specification.")
elif page=="Evaluation":
    st.markdown("<div class='kicker'>TEST PERFORMANCE</div>",unsafe_allow_html=True); st.title("Evaluation")
    names=list(MODEL_RESULTS)
    st.bar_chart({"Accuracy (%)":{n:MODEL_RESULTS[n]["accuracy"] for n in names},"Macro F1 (%)":{n:MODEL_RESULTS[n]["f1"] for n in names}})
    st.markdown("### Hybrid model confusion matrix"); st.write("Rows = actual class · Columns = predicted class")
    cm=pd.DataFrame([[316,0,0],[1,302,5],[1,8,333]],index=CLASSES,columns=CLASSES)
    st.dataframe(cm,use_container_width=True)
    st.caption("Recorded test set: 966 images; 951 correct predictions (98.45%). Verify against the final notebook.")
elif page=="Dataset & Methodology":
    st.markdown("<div class='kicker'>DATA & EXPERIMENT DESIGN</div>",unsafe_allow_html=True); st.title("Dataset & Methodology")
    a,b,c=st.columns(3); a.metric("Selected images","6,436"); b.metric("Training","4,505"); c.metric("Validation + test","1,931")
    st.markdown("### Class distribution")
    st.bar_chart(pd.DataFrame({"Images":[2276,2103,2057]},index=["Salmo","Cocci","Healthy"]))
    st.markdown("### Workflow")
    st.markdown("1. Select and label three target classes.\n2. Split into training, validation and test subsets.\n3. Train and compare transfer-learning models.\n4. Evaluate accuracy, precision, recall, macro F1 and confusion matrix.\n5. Run inference with the saved Hybrid Model.")
    st.warning("Verify dataset counts against the final notebook before publication.")
