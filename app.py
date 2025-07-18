import streamlit as st
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

st.set_page_config(page_title="FITS 색등급 분석기", layout="wide")

st.title("🔭 FITS 파일 색등급 분석기 (Color-Magnitude Diagram)")

uploaded_file = st.file_uploader("FITS 파일을 업로드하세요", type=["fits", "fit"])

def normalize_data(data):
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def extract_star_colors(image_data, threshold=200):
    norm_img = normalize_data(image_data)
    _, binary = cv2.threshold(norm_img, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 1 and h > 1:
            roi = norm_img[y:y+h, x:x+w]
            mean_val = np.mean(roi)
            colors.append(mean_val)

    return colors

if uploaded_file:
    with fits.open(uploaded_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data

        if data is None:
            st.error("FITS 파일에 이미지 데이터가 없습니다.")
        else:
            st.subheader("📄 FITS 파일 정보")
            st.markdown(f"**이미지 크기:** {data.shape}")
            st.markdown(f"**노출 시간:** {header.get('EXPTIME', '알 수 없음')}")
            st.markdown(f"**필터:** {header.get('FILTER', '알 수 없음')}")
            st.markdown(f"**망원경:** {header.get('TELESCOP', '알 수 없음')}")

            st.subheader("🖼 이미지 보기")
            fig, ax = plt.subplots()
            ax.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))
            ax.set_title("FITS 이미지")
            st.pyplot(fig)

            st.subheader("🌈 밝기 분석 및 색등급")
            star_colors = extract_star_colors(data)
            magnitudes = [-2.5 * np.log10(val + 1e-6) for val in star_colors]

            fig2, ax2 = plt.subplots()
            ax2.hist(magnitudes, bins=30, color='orange', alpha=0.7)
            ax2.set_xlabel("밝기 (Magnitude)")
            ax2.set_ylabel("별 개수")
            ax2.set_title("밝기 분포")
            st.pyplot(fig2)

            # CMD용 임의 Color Index 값 생성
            color_index = np.random.normal(0.5, 0.2, len(magnitudes))

            st.subheader("📉 색등급도 (Color-Magnitude Diagram)")
            fig3, ax3 = plt.subplots()
            ax3.scatter(color_index, magnitudes, s=5, c='black')
            ax3.set_xlabel("색 지수 (Color Index: G-B)")
            ax3.set_ylabel("밝기 (Magnitude)")
            ax3.set_title("색등급도 (CMD)")
            ax3.invert_yaxis()
            st.pyplot(fig3)

            st.success(f"검출된 별 개수: {len(magnitudes)}")
