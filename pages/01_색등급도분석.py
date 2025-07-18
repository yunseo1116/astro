import streamlit as st
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="FITS 색등급 분석기", layout="wide")
st.title("🌟 성단 색등급도 분석기 (CMD)")

st.markdown("두 개의 필터(G, B)로 찍은 FITS 파일을 업로드하세요.")

g_file = st.file_uploader("G 필터 이미지 (예: green 필터)", type=["fits", "fit"], key="g")
b_file = st.file_uploader("B 필터 이미지 (예: blue 필터)", type=["fits", "fit"], key="b")

def normalize_data(data):
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def extract_star_positions(image_data, threshold=200):
    norm_img = normalize_data(image_data)
    _, binary = cv2.threshold(norm_img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        if w > 1 and h > 1:
            positions.append((cx, cy))
    return positions

def extract_brightness(data, positions, size=5):
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
        g_data = g_hdul[0].data
        b_data = b_hdul[0].data

        if g_data is None or b_data is None:
            st.error("FITS 파일에 이미지 데이터가 없습니다.")
        else:
            g_data = np.nan_to_num(g_data)
            b_data = np.nan_to_num(b_data)

            st.subheader("📈 이미지 시각화")
            fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
            ax1[0].imshow(g_data, cmap='gray', origin='lower', vmin=np.percentile(g_data, 5), vmax=np.percentile(g_data, 99))
            ax1[0].set_title("G 필터")
            ax1[1].imshow(b_data, cmap='gray', origin='lower', vmin=np.percentile(b_data, 5), vmax=np.percentile(b_data, 99))
            ax1[1].set_title("B 필터")
            st.pyplot(fig1)

            st.subheader("🌟 별 검출 및 색지수 계산")
            positions = extract_star_positions(g_data, threshold=200)
            g_brightness = extract_brightness(g_data, positions)
            b_brightness = extract_brightness(b_data, positions)

            # 로그 스케일로 변환: Magnitude
            g_mag = [-2.5 * np.log10(val + 1e-6) for val in g_brightness]
            b_mag = [-2.5 * np.log10(val + 1e-6) for val in b_brightness]
            color_index = [b - g for b, g in zip(b_mag, g_mag)]

            st.write(f"총 검출된 별 개수: {len(color_index)}")

            st.subheader("📉 색등급도 (CMD)")
            fig2, ax2 = plt.subplots()
            ax2.scatter(color_index, g_mag, s=5, c='blue')
            ax2.set_xlabel("색지수 (B − G)")
            ax2.set_ylabel("밝기 (G Mag)")
            ax2.set_title("Color-Magnitude Diagram (CMD)")
            ax2.invert_yaxis()
            st.pyplot(fig2)

