test_cases = [
    {
        "query": "Khoản hoàn tiền được xử lý trong bao lâu và tính từ mốc nào?",
        "expected_facts": [
            "5 ngày làm việc",
            "ngày phương án hoàn tiền được chấp thuận"
        ],
        "expected_source": "sample.pdf",
        "expected_page": 8
    },
    {
        "query": "Đơn nội thành xác nhận đúng 15 giờ bắt đầu tính thời gian giao từ ngày nào?",
        "expected_facts": [
            "ngày làm việc kế tiếp"
        ],
        "expected_source": "sample.pdf",
        "expected_page": 3
    },
    {
        "query": "Địa chỉ email hỗ trợ của cửa hàng là gì?",
        "expected_facts": [
            "hotro@cuahangmau.example"
        ],
        "expected_source": "sample.pdf",
        "expected_page": 13
    },
    {
        "query": "Số điện thoại hỗ trợ của cửa hàng là gì?",
        "expected_facts": [
            "không được công bố"
        ],
        "expected_source": "sample.pdf",
        "expected_page": 13
    }
]