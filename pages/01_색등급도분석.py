import streamlit as st
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass

st.set_page_config(page_title="성단 색등급도 분석기", layout="wide")
st.title("🌟 성단 색등급도 분석기 (CMD)")

st.markdown("두 개의 필터(G, B)로 찍은 FITS 파일을 업로드하세요.")

g_file = st.file_uploader("G 필터 이미지 (예: green 필터)", type=["fits", "fit"], key="g")
b_file = st.file_uploader("B 필터 이미지 (예: blue 필터)", type=["fits", "fit"], key="b")

def detect_star_positions(data, threshold_percentile=99.5):
    """밝은 점(별) 위치 검출"""
    threshold = np.percentile(data, threshold_percentile)
    binary = data > threshold
    labeled, _ = label(binary)
    positions = center_of_mass(data, labeled, range(1, np.max(labeled)+1))
    return [(int(x), int(y)) for y, x in positions]

def extract_brightness(data, positions, size=5):
    """중심점 주변 밝기 총합"""
    brightness = []
    for x, y in positions:
        x1, x2 = max(0, x-size), min(data.shape[1], x+size)
        y1, y2 = max(0, y-size), min(data.shape[0], y+size)
        roi = data[y1:y2, x1:x2]
        val = np.sum(roi)
        brightness.append(val)
    return brightness

if g_file and b_file:
    with fits.open(g_file) as g_hdul, fits.open(b_file) as b_hdul:
        g_data = np.nan_to_num(g_hdul[0].data)
        b_data = np.nan_to_num(b_hdul[0].data)

        st.subheader("📈 이미지 시각화")
        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        ax1[0].imshow(g_data, cmap='gray', origin='lower', vmin=np.percentile(g_data, 5), vmax=np.percentile(g_data, 99))
        ax1[0].set_title("G 필터")
        ax1[1].imshow(b_data, cmap='gray', origin='lower', vmin=np.percentile(b_data, 5), vmax=np.percentile(b_data, 99))
        ax1[1].set_title("B 필터")
        st.pyplot(fig1)

        st.subheader("🔎 별 검출 및 밝기 계산")
        positions = detect_star_positions(g_data, threshold_percentile=99.5)
        st.write(f"검출된 별 수: {len(positions)}")

        g_brightness = extract_brightness(g_data, positions)
        b_brightness = extract_brightness(b_data, positions)

        g_mag = [-2.5 * np.log10(val + 1e-6) for val in g_brightness]
        b_mag = [-2.5 * np.log10(val + 1e-6) for val in b_brightness]
        color_index = [b - g for b, g in zip(b_mag, g_mag)]

        st.subheader("📉 색등급도 (CMD)")
        fig2, ax2 = plt.subplots()
        ax2.scatter(color_index, g_mag, s=5, c='blue', alpha=0.7)
        ax2.set_xlabel("색지수 (B − G)")
        ax2.set_ylabel("밝기 (G Magnitude)")
        ax2.set_title("Color-Magnitude Diagram (CMD)")
        ax2.invert_yaxis()
        st.pyplot(fig2)
