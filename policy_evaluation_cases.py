test_cases = [
    {
        "query": "Điều kiện để đổi sản phẩm trong 14 ngày là gì?",
        "answerable": True,
        "expected_fact": "mã đơn và hóa đơn"
    },
    {
        "query": "Sản phẩm không bị lỗi nhưng tôi muốn đổi sang mẫu khác thì có được không?",
        "answerable": True,
        "expected_fact": "Không áp dụng đổi sản phẩm"
    },
    {
        "query": "Phí vận chuyển khi gửi sản phẩm đi bảo hành là bao nhiêu?",
        "answerable": False,
        "expected_fact": None
    },
    {
        "query": "Đơn hàng nội thành Hà Nội được giao trong bao lâu kể từ ngày xác nhận?",
        "answerable": True,
        "expected_fact": "2 ngày làm việc"
    },
    {
        "query": "Sau khi phương án hoàn tiền được chấp thuận, khoản hoàn tiền được xử lý trong bao lâu?",
        "answerable": True,
        "expected_fact": "5 ngày làm việc"
    },
    {
        "query": "Khi gửi yêu cầu bảo hành, khách hàng cần cung cấp thông tin gì để đối chiếu thời hạn và sản phẩm?",
        "answerable": True,
        "expected_fact": "mã đơn hoặc hóa đơn điện tử"
    },
    {
        "query": "Trung tâm tiếp nhận sản phẩm từ mấy giờ đến mấy giờ?",
        "answerable": True,
        "expected_fact": "8 giờ 30 đến 17 giờ 30"
    },
    {
        "query": "Số điện thoại hỗ trợ của cửa hàng là số nào?",
        "answerable": False,
        "expected_fact": None
    }
]