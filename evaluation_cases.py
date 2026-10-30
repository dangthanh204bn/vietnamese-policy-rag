test_cases = [
    {
        "query": "Sản phẩm được bảo hành bao lâu?",
        "answerable": True,
        "expected_fact": "18 tháng"
    },
    {
        "query": "Đổi sản phẩm trong bao lâu?",
        "answerable": True,
        "expected_fact": "14 ngày"

    },
    {
        "query": "Giao hàng trong bao lâu?",
        "answerable": True,
        "expected_fact": "2 ngày"
    },
    {
        "query": "Dung lượng pin là bao nhiêu?",
        "answerable": False,
        "expected_fact": None
    },
    {
        "query": "Địa chỉ bảo hành ở đâu?",
        "answerable": False,
        "expected_fact": None
    },
    {
        "query": "Sản phẩm được bảo hành tại địa chỉ nào?",
        "answerable": False,
        "expected_fact": None
    }

]