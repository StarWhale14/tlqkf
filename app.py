import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import zipfile
from datetime import datetime

# -----------------------------
# 기본 설정값
# -----------------------------
DEFAULT_TOLERANCE = 15
DEFAULT_SV_MIN_SAT = 40
DEFAULT_SV_MIN_VAL = 40
DEFAULT_LINE_THICKNESS = 2


# -----------------------------
# 색상 추출 함수
# -----------------------------
def extract_color_area(pil_img, selected_rgb, hue_tolerance, thickness,
                       sv_min_sat=DEFAULT_SV_MIN_SAT, sv_min_val=DEFAULT_SV_MIN_VAL,
                       auto_thickness=False):
    pil_img = pil_img.convert("RGBA")
    src = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
    rgb = src[:, :, :3]
    alpha = src[:, :, 3]

    # RGB → HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sel_arr = np.uint8([[[selected_rgb[0], selected_rgb[1], selected_rgb[2]]]])
    sel_hsv = cv2.cvtColor(sel_arr, cv2.COLOR_RGB2HSV)[0, 0]
    sel_h = int(sel_hsv[0])

    low_h = sel_h - hue_tolerance
    high_h = sel_h + hue_tolerance

    if low_h < 0:
        lower1 = np.array([0, sv_min_sat, sv_min_val])
        upper1 = np.array([high_h, 255, 255])
        lower2 = np.array([180 + low_h, sv_min_sat, sv_min_val])
        upper2 = np.array([179, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
    elif high_h > 179:
        lower1 = np.array([low_h, sv_min_sat, sv_min_val])
        upper1 = np.array([179, 255, 255])
        lower2 = np.array([0, sv_min_sat, sv_min_val])
        upper2 = np.array([high_h - 180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
    else:
        lower = np.array([low_h, sv_min_sat, sv_min_val])
        upper = np.array([high_h, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    # 알파 마스크 적용
    mask = cv2.bitwise_and(mask, np.where(alpha > 10, 255, 0).astype(np.uint8))

    if auto_thickness:
        h_img, w_img = mask.shape
        longest = max(h_img, w_img)
        thickness = max(1, min(20, int(longest / 400)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (max(1, thickness), max(1, thickness)))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    if thickness > 1:
        clean = cv2.dilate(clean, kernel, iterations=1)

    # 결과 생성 (투명 배경 유지)
    result_rgb = np.where(clean[..., None] > 0, rgb, 0).astype(np.uint8)
    result_alpha = np.where(clean > 0, 255, 0).astype(np.uint8)
    result = np.dstack((result_rgb, result_alpha))
    return Image.fromarray(result)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="HSV 색상 추출기", layout="wide")
st.title("🎨 HSV 기반 색상 추출기")

uploaded_files = st.file_uploader(
    "이미지 파일 선택 (여러 개 가능)",
    type=["png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

# 색상 선택
selected_color = st.color_picker("🎨 추출할 색상 선택", "#ff0000")
selected_rgb = tuple(int(selected_color[i:i + 2], 16) for i in (1, 3, 5))

# 슬라이더 UI
col1, col2, col3 = st.columns(3)
with col1:
    tolerance = st.slider("색상 범위 (Hue ±)", 0, 90, DEFAULT_TOLERANCE)
with col2:
    min_sat = st.slider("최소 채도", 0, 255, DEFAULT_SV_MIN_SAT)
with col3:
    min_val = st.slider("최소 명도", 0, 255, DEFAULT_SV_MIN_VAL)

col4, col5 = st.columns(2)
with col4:
    thickness = st.slider("두께 조절", 1, 20, DEFAULT_LINE_THICKNESS)
with col5:
    auto_thick = st.checkbox("자동 두께 조절", value=False)

# 배경색
bg_color = st.radio("미리보기 배경색", ["White", "Gray", "Black"], horizontal=True)
bg_rgba = {"White": (255, 255, 255, 255),
           "Gray": (128, 128, 128, 255),
           "Black": (0, 0, 0, 255)}[bg_color]


# -----------------------------
# 처리 버튼
# -----------------------------
if st.button("💾 색상 추출 시작"):
    if not uploaded_files:
        st.warning("먼저 이미지를 업로드하세요!")
    else:
        result_zip = io.BytesIO()
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in uploaded_files:
                pil_img = Image.open(file)
                result_img = extract_color_area(
                    pil_img, selected_rgb, tolerance, thickness,
                    sv_min_sat=min_sat, sv_min_val=min_val,
                    auto_thickness=auto_thick
                )

                bg_img = Image.new("RGBA", result_img.size, bg_rgba)
                bg_img.alpha_composite(result_img)

                # Streamlit 렌더링 호환용
                st.image(np.array(pil_img.convert("RGB")),
                         caption=f"원본: {file.name}", use_container_width=True)
                st.image(np.array(bg_img.convert("RGB")),
                         caption=f"추출 결과: {file.name}", use_container_width=True)

                # ZIP 저장
                buf = io.BytesIO()
                bg_img.save(buf, format="PNG")
                base_name = file.name.rsplit(".", 1)[0]
                zipf.writestr(f"{base_name}_extract.png", buf.getvalue())

        result_zip.seek(0)
        st.download_button(
            "📦 ZIP으로 다운로드",
            result_zip,
            file_name=f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )

st.caption("© 2025 HSV 색상 추출 웹앱 (안정 버전)")