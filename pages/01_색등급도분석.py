import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="별 색으로 표면 온도 계산기")

st.title("⭐ 별 색으로 표면 온도 계산하기")

uploaded_img = st.file_uploader("별 사진(JPG/PNG)을 올려주세요", type=['jpg','jpeg','png'])

if uploaded_img:
    img = Image.open(uploaded_img).convert('RGB')
    st.image(img, caption="업로드된 별 사진", use_container_width=True)
    
    coords = streamlit_image_coordinates(img, key="coords")
    
    if coords is not None:
        x, y = int(coords['x']), int(coords['y'])
        
        if 0 <= x < img.width and 0 <= y < img.height:
            r, g, b = img.getpixel((x, y))
            
            # B-V 근사 계산
            bv = max(0.0, min(2.0, (0.3 * (b - r)) / 255 + 0.65))
            
            # Ballesteros 공식에 따른 표면 온도 계산 (켈빈)
            temperature = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))
            
            st.markdown("### 선택한 위치 정보")
            st.write(f"📍 좌표: (x={x}, y={y})")
            st.write(f"🎨 RGB 값: R={r}, G={g}, B={b}")
            st.write(f"🌈 근사 B−V 색지수: {bv:.2f}")
            st.write(f"🔥 추정 별 표면 온도: {temperature:.0f} K")
        else:
            st.warning("클릭한 좌표가 이미지 범위를 벗어났습니다.")
else:
    st.info("별 사진 파일을 업로드해 주세요.")
