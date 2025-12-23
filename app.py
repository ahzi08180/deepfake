# 左右併排並置中
col1, col2, col3 = st.columns([1,3,1])  # 左空白、內容、右空白

with col1:
    st.write("")  # 空白

with col2:
    # 再用內部兩列左右併排圖片與結果
    inner_col1, inner_col2 = st.columns([2,1])
    
    with inner_col1:
        st.image(img_pil, caption="Detected Face", use_column_width=True)

    with inner_col2:
        st.markdown(f"""
        <div style="padding:20px; border-radius:15px; text-align:center;">
            <h2 style="color:#e63946; font-size:24px;">Fake Probability</h2>
            <h1 style="color:#e63946; font-size:48px;">{p*100:.2f}%</h1>
        </div>

        <div style="position: relative; width:200px; height:200px; margin:auto; margin-top:10px;">
            <svg viewBox="0 0 36 36" class="circular-chart">
                <path class="circle-bg"
                    d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0-31.831"/>
                <path class="circle"
                    stroke-dasharray="{p*100}, 100"
                    d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0-31.831"/>
                <text x="18" y="20.35" class="percentage">{p*100:.1f}%</text>
            </svg>
        </div>

        <style>
        .circular-chart {{ display:block; max-width:100%; max-height:100%; }}
        .circle-bg {{ fill:none; stroke:#eee; stroke-width:5; }}
        .circle {{ fill:none; stroke:#e63946; stroke-width:5; stroke-linecap:round; transition: stroke-dasharray 0.3s; }}
        .percentage {{ fill:#e63946; font-size:1em; text-anchor:middle; }}
        </style>
        """, unsafe_allow_html=True)

with col3:
    st.write("")  # 空白
