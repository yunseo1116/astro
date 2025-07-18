import streamlit as st
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass

st.set_page_config(page_title="성단 B–V 색등급도 분석기", layout="wide")
st.title("🌌 B–V 필터 기반 색등급도 분석")

st.markdown("Messier 12 같은 성단의 **B 필터**와 **V 필터** FITS 파일을 업로드하세요.")

b_file = st.file_uploader("B 필터 FITS 파일", type=["fits", "fit"])
v_file = st.file_uploader("V 필터 FITS 파일", type=["fits", "fit"])

def detect_star_positions(data, perc=99.5):
    thr = np.percentile(data, perc)
    mask = data > thr
    labeled, _ = label(mask)
    coords = center_of_mass(data, labeled, range(1, np.max(labeled)+1))
    return [(int(x), int(y)) for y, x in coords]

def measure_brightness(data, positions, radius=5):
    vals = []
    for x, y in positions:
        x1, x2 = max(0, x-radius), min(data.shape[1], x+radius)
        y1, y2 = max(0, y-radius), min(data.shape[0], y+radius)
        vals.append(np.sum(data[y1:y2, x1:x2]))
    return vals

if b_file and v_file:
    b_data = np.nan_to_num(fits.open(b_file)[0].data)
    v_data = np.nan_to_num(fits.open(v_file)[0].data)

    st.subheader("📷 이미지 미리보기")
    f, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(b_data, cmap='gray', origin='lower')
    ax[0].set_title("B 필터")
    ax[1].imshow(v_data, cmap='gray', origin='lower')
    ax[1].set_title("V 필터")
    st.pyplot(f)

    positions = detect_star_positions(v_data)
    st.write(f"📍 검출된 별 수: {len(positions)}")

    b_vals = measure_brightness(b_data, positions)
    v_vals = measure_brightness(v_data, positions)

    b_mag = [-2.5*np.log10(v+1e-6) for v in b_vals]
    v_mag = [-2.5*np.log10(v+1e-6) for v in v_vals]
    color_index = [b-m for b,m in zip(b_mag, v_mag)]

    st.subheader("📊 색등급도 (CMD: B–V vs V)")
    fig2, ax2 = plt.subplots()
    ax2.scatter(color_index, v_mag, s=5, c='purple', alpha=0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("색지수 (B – V)")
    ax2.set_ylabel("V 필터 밝기 (Magnitude)")
    st.pyplot(fig2)
