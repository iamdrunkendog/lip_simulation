import streamlit as st
import numpy as np
import json
import os

from render_once import render_lips

# =====================================================
# Page
# =====================================================
st.set_page_config(layout="wide")
st.title("Lip Gloss Renderer – Tuning GUI")

# =====================================================
# Constants / Directories
# =====================================================
PRESET_DIR = "presets"
os.makedirs(PRESET_DIR, exist_ok=True)

TEST_IMAGE_DIR = "test_images"
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)

PRESET_KEYS = [
    # Mask / Geometry
    "EDGE_UPPER", "EDGE_LOWER", "MIDLINE_BIAS",

    # Color
    "LIP_COLOR_HEX", "COLOR_OPACITY",

    # Fake Normal
    "LIQUID_BLUR_SIGMA", "HEIGHT_GAIN",
    "HIGH_BLUR_SIGMA", "HIGH_GAIN", "ALPHA",

    # Specular
    "SPEC_STRENGTH", "SHININESS",

    # Clear Coat
    "CLEAR_STRENGTH", "CLEAR_EXPONENT",

    # Highlight Shape
    "UPPER_CENTER", "LOWER_CENTER",

    # Light
    "LIGHT_DIR",
]

# =====================================================
# Helpers
# =====================================================
def safe_light_dir(v):
    """
    Always returns a (3,) float32 numpy vector.
    Protects specular.py from "iteration over a 0-d array".
    """
    v = np.asarray(v, np.float32)

    # scalar / 0-d
    if v.ndim == 0:
        return np.array([0.0, -0.25, 1.0], np.float32)

    # wrong shape
    if v.shape != (3,):
        return np.array([0.0, -0.25, 1.0], np.float32)

    return v


def export_preset(name: str):
    """
    Save only known preset keys (JSON-safe).
    Convert numpy arrays to lists.
    """
    preset = {}
    for k in PRESET_KEYS:
        v = st.session_state.get(k)

        if isinstance(v, np.ndarray):
            preset[k] = v.tolist()
        else:
            preset[k] = v

    path = os.path.join(PRESET_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2, ensure_ascii=False)

    return path


def load_preset(path: str):
    """
    Load preset keys into session_state safely.
    Restore LIGHT_DIR as (3,) float32 numpy vector.
    """
    with open(path, "r", encoding="utf-8") as f:
        preset = json.load(f)

    for k, v in preset.items():
        if k == "LIGHT_DIR":
            st.session_state[k] = safe_light_dir(v)
        else:
            st.session_state[k] = v


# =====================================================
# Sidebar – Presets
# =====================================================
st.sidebar.header("Presets")

preset_name = st.sidebar.text_input("Preset name", value="my_preset")

if st.sidebar.button("Save Preset"):
    export_preset(preset_name)
    st.sidebar.success(f"Saved preset: {preset_name}")

preset_files = [
    f.replace(".json", "")
    for f in os.listdir(PRESET_DIR)
    if f.endswith(".json")
]

selected_preset = st.sidebar.selectbox(
    "Load preset",
    options=[""] + preset_files
)

if selected_preset:
    load_preset(os.path.join(PRESET_DIR, f"{selected_preset}.json"))
    st.sidebar.success(f"Loaded preset: {selected_preset}")

st.sidebar.divider()

# =====================================================
# Sidebar – Test Image
# =====================================================
st.sidebar.header("Test Image")

test_images = sorted([
    f for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not test_images:
    st.sidebar.warning(f"No test images found in '{TEST_IMAGE_DIR}'")
    st.sidebar.caption("Add .jpg/.png files into test_images/ and rerun.")
    st.stop()

# Initialize current test image
if "test_image" not in st.session_state:
    st.session_state.test_image = os.path.join(TEST_IMAGE_DIR, test_images[0])

# If current image is missing (deleted/renamed), fallback
current_basename = os.path.basename(st.session_state.test_image)
if current_basename not in test_images:
    st.session_state.test_image = os.path.join(TEST_IMAGE_DIR, test_images[0])
    current_basename = os.path.basename(st.session_state.test_image)

selected_image = st.sidebar.selectbox(
    "Select test image",
    options=test_images,
    index=test_images.index(current_basename)
)

# Update selected image
st.session_state.test_image = os.path.join(TEST_IMAGE_DIR, selected_image)
st.sidebar.divider()

# =====================================================
# Sidebar – Parameters
# =====================================================
st.sidebar.header("Parameters")

with st.sidebar.expander("Color", expanded=True):
    # Default to a red-ish color
    default_color = st.session_state.get("LIP_COLOR_HEX", "#D01020")
    
    LIP_COLOR_HEX = st.color_picker(
        "Lip Color", 
        value=default_color,
        key="LIP_COLOR_HEX",
        help="원하는 립 컬러를 선택하세요."
    )

    COLOR_OPACITY = st.slider(
        "COLOR_OPACITY", 0.0, 1.0,
        st.session_state.get("COLOR_OPACITY", 0.8),
        help="립스틱 색이 입술색을 덮는 정도입니다."
    )


# ------------------
# Mask / Geometry
# ------------------
with st.sidebar.expander("Mask / Geometry", expanded=False):
    EDGE_UPPER = st.slider(
        "EDGE_UPPER", 1, 20,
        st.session_state.get("EDGE_UPPER", 6), 1,
        key="EDGE_UPPER",
        help="윗입술 가장자리의 마스크 경계를 얼마나 안쪽/바깥쪽으로 잡을지 조절합니다. 립 컬러의 경계 부드러움(fuzziness)도 함께 조절됩니다."
    )

    EDGE_LOWER = st.slider(
        "EDGE_LOWER", 1, 20,
        st.session_state.get("EDGE_LOWER", 8), 1,
        key="EDGE_LOWER",
        help="아랫입술 가장자리의 마스크 두께 및 경계 강도를 조절합니다. 립 컬러의 경계 부드러움(fuzziness)도 함께 조절됩니다."
    )

    MIDLINE_BIAS = st.slider(
        "MIDLINE_BIAS", 0.0, 0.3,
        st.session_state.get("MIDLINE_BIAS", 0.12), 0.01,
        key="MIDLINE_BIAS",
        help="윗입술과 아랫입술의 경계선 위치를 위/아래로 미세 조정합니다."
    )


# ------------------
# Fake Normal
# ------------------
with st.sidebar.expander("Fake Normal", expanded=True):
    LIQUID_BLUR_SIGMA = st.slider(
        "LIQUID_BLUR_SIGMA", 1, 60,
        st.session_state.get("LIQUID_BLUR_SIGMA", 18), 1,
        key="LIQUID_BLUR_SIGMA",
        help="립 표면을 얼마나 액체처럼 부드럽게 퍼뜨릴지 결정합니다. 값이 클수록 매끈해집니다."
    )

    HEIGHT_GAIN = st.slider(
        "HEIGHT_GAIN", 0.0, 20.0,
        st.session_state.get("HEIGHT_GAIN", 8.0), 0.5,
        key="HEIGHT_GAIN",
        help="입술 표면의 볼륨감(도톰함)을 조절합니다."
    )

    HIGH_BLUR_SIGMA = st.slider(
        "HIGH_BLUR_SIGMA", 0.1, 10.0,
        st.session_state.get("HIGH_BLUR_SIGMA", 2.0), 0.1,
        key="HIGH_BLUR_SIGMA",
        help="하이라이트 영역의 퍼짐 정도를 조절합니다."
    )

    HIGH_GAIN = st.slider(
        "HIGH_GAIN", 0.0, 10.0,
        st.session_state.get("HIGH_GAIN", 3.0), 0.1,
        key="HIGH_GAIN",
        help="광택 하이라이트의 강도를 조절합니다."
    )

    ALPHA = st.slider(
        "ALPHA", 0.0, 1.0,
        st.session_state.get("ALPHA", 0.25), 0.01,
        key="ALPHA",
        help="전체 립 효과의 혼합 비율입니다. 원본과 합성의 밸런스를 조절합니다."
    )


# ------------------
# Specular
# ------------------
with st.sidebar.expander("Specular", expanded=True):
    SPEC_STRENGTH = st.slider(
        "SPEC_STRENGTH", 0.0, 0.5,
        st.session_state.get("SPEC_STRENGTH", 0.18), 0.01,
        key="SPEC_STRENGTH",
        help="직접 반사광의 세기입니다. 글로시한 느낌에 가장 큰 영향을 줍니다."
    )

    SHININESS = st.slider(
        "SHININESS", 10, 200,
        st.session_state.get("SHININESS", 60), 1,
        key="SHININESS",
        help="하이라이트의 날카로움입니다. 높을수록 작은 영역에 강한 반짝임이 생깁니다."
    )


# ------------------
# Clear Coat
# ------------------
with st.sidebar.expander("Clear Coat", expanded=True):
    CLEAR_STRENGTH = st.slider(
        "CLEAR_STRENGTH", 0.0, 0.6,
        st.session_state.get("CLEAR_STRENGTH", 0.4), 0.01, key="CLEAR_STRENGTH"
    )
    CLEAR_EXPONENT = st.slider(
        "CLEAR_EXPONENT", 5, 120,
        st.session_state.get("CLEAR_EXPONENT", 30), 1, key="CLEAR_EXPONENT"
    )

# ------------------
# Highlight Shape
# ------------------
with st.sidebar.expander("Highlight Shape", expanded=True):
    UPPER_CENTER = st.slider(
        "UPPER_CENTER", -0.9, -0.1,
        st.session_state.get("UPPER_CENTER", -0.8), 0.01, key="UPPER_CENTER"
    )
    LOWER_CENTER = st.slider(
        "LOWER_CENTER", 0.0, 1.0,
        st.session_state.get("LOWER_CENTER", 0.55), 0.01, key="LOWER_CENTER"
    )

# =====================================================
# Params dict
# =====================================================
params = {
    
    
    "EDGE_UPPER": EDGE_UPPER,
    "EDGE_LOWER": EDGE_LOWER,
    "MIDLINE_BIAS": MIDLINE_BIAS,

    "LIQUID_BLUR_SIGMA": LIQUID_BLUR_SIGMA,
    "HEIGHT_GAIN": HEIGHT_GAIN,
    "HIGH_BLUR_SIGMA": HIGH_BLUR_SIGMA,
    "HIGH_GAIN": HIGH_GAIN,
    "ALPHA": ALPHA,

    "SPEC_STRENGTH": SPEC_STRENGTH,
    "SHININESS": SHININESS,

    # IMPORTANT: Always coerce to (3,) float32
    "LIGHT_DIR": safe_light_dir(
        st.session_state.get("LIGHT_DIR", [0.0, -0.25, 1.0])
    ),

    "CLEAR_STRENGTH": CLEAR_STRENGTH,
    "CLEAR_EXPONENT": CLEAR_EXPONENT,

    "UPPER_CENTER": UPPER_CENTER,
    "LOWER_CENTER": LOWER_CENTER,

    "COLOR": {
        "R": int(LIP_COLOR_HEX[1:3], 16),
        "G": int(LIP_COLOR_HEX[3:5], 16),
        "B": int(LIP_COLOR_HEX[5:7], 16),
        "OPACITY": COLOR_OPACITY,
    },
    
}

# =====================================================
# Render
# =====================================================
results = render_lips(st.session_state.test_image, params)

# =====================================================
# Display
# =====================================================
st.header("Zoomed Comparison")
zcol1, zcol2 = st.columns(2)

with zcol1:
    st.subheader("Zoomed Original")
    st.image(results["original_roi"], channels="BGR", use_container_width=True)

with zcol2:
    st.subheader("Zoomed Final")
    st.image(results["final_roi"], channels="BGR", use_container_width=True)

st.divider()

st.header("Full Face Comparison")
col1, col2, col3 = st.columns(3)

with col3:
    st.subheader("Final")
    st.image(results["final"], channels="BGR", use_container_width=True)

with col2:
    st.subheader("Original")
    st.image(results["original"], channels="BGR", use_container_width=True)

with col1:
    st.subheader("Debug Maps")

    with st.expander("Mask / Geometry", expanded=False):
        for k in ["lip01", "x_hat", "y_hat", "r"]:
            st.markdown(f"**{k}**")
            st.image(results[k], width=500)

    with st.expander("Fake Normal", expanded=True):
        for k in ["normal_Nx", "normal_Ny", "normal_Nz", "normal_length"]:
            st.markdown(f"**{k}**")
            st.image(results[k], width=500)

    with st.expander("Highlight Shape", expanded=True):
        st.image(results["hwf"], width=500)

    with st.expander("Specular", expanded=False):
        st.image(results["spec"], width=500)

    with st.expander("Clear Coat", expanded=True):
        st.image(results["clear_weight"], width=500)
        st.image(results["clear_spec"], width=500)
