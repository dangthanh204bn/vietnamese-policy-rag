import streamlit as st
import requests

if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

st.title("Hỏi đáp tài liệu")
uploaded_file = st.file_uploader("Chọn tài liệu PDF", type=["pdf"])

if st.button("Nạp tài liệu"):
    if uploaded_file is None:
        st.warning("Chưa chọn PDF")
    else:
        try:
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }
            uploaded_response = requests.post(url="http://127.0.0.1:8000/upload", files=files, timeout=180)
            uploaded_response.raise_for_status()
            st.session_state["last_response"] = None
            st.write("File", uploaded_file.name)
            st.write("Kích thước", uploaded_file.size)
            upload_data = uploaded_response.json()
            st.success(f"Tên tài liệu: {upload_data["filename"]}")
            st.success(f"Số chunk từ: {upload_data["chunk_count"]}")
        except requests.exceptions.HTTPError:
            st.error("API không nạp được tài liệu.")
        except requests.exceptions.Timeout:
            st.error("Hết thời gian chờ nạp tài liệu.")
        except requests.exceptions.ConnectionError:
            st.error("Không kết nối được API.")

query = st.text_input("Nhập câu hỏi")
if st.button("Gửi câu hỏi"):
    st.session_state["last_response"] = None
    try:
        response = requests.post(url="http://127.0.0.1:8000/ask", json={"query": query}, timeout=180)
        response.raise_for_status()
        data = response.json()
        st.session_state["last_response"] = data
        
    
    except requests.exceptions.HTTPError:
        detail = response.json().get("detail", "Yêu cầu API thất bại")
        st.error(detail)

    except requests.exceptions.Timeout:
        st.error("Hết thời gian chờ phản hồi từ API.")
    except requests.exceptions.ConnectionError:
        st.error("Không kết nối được API. Hãy kiểm tra FastAPI đang chạy.")

if st.session_state["last_response"] is not None:
    data = st.session_state["last_response"]
    st.write(data["received_query"])
    st.write(data["answer"])
    for source in data["retrieved_sources"]:
        page_number = source.get("page_number")
        if page_number != None:
            label = f"[chunk_{source['chunk_id']}] {source['source']} | Trang: {page_number}"
        else:
            label = f"[chunk_{source['chunk_id']}] {source['source']}"


        with st.expander(label):
            st.write(source["text"])