import random
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_test_records(n=100):
    firm_names = ["Tesla", "BYD", "Apple", "Google", "Amazon", "Meta", "NIO", "Samsung", "IBM", "Intel"]
    base_month = datetime(2025, 5, 1)  # 设置起始月份

    dimension_templates = {
        "patent": [
            "The firm secured multiple patents for its latest energy innovation.",
            "New intellectual property filings highlight strong R&D activity.",
            "Innovation breakthroughs led to patent awards in AI technology.",
            "They registered several patents this month for chip manufacturing.",
            "Patent activity surged in the field of autonomous vehicles."
        ],
        "reputation": [
            "Customer satisfaction ratings improved significantly this quarter.",
            "The firm faces backlash due to data privacy concerns.",
            "Brand trust rose after successful product recalls and fixes.",
            "Analysts report a favorable shift in public perception.",
            "Negative reviews impacted the company’s online image."
        ],
        "executive": [
            "The CEO unveiled a five-year global expansion roadmap.",
            "Top leadership was reshuffled to drive innovation.",
            "Executive board committed to sustainability transitions.",
            "A new strategic office was launched under executive leadership.",
            "Leaders pledged investment in digital transformation."
        ],
        "crypto": [
            "They announced a new blockchain payment protocol.",
            "The firm adopted Ethereum-based smart contracts.",
            "Digital tokens will be used for internal transactions.",
            "Partnership with crypto exchanges was finalized.",
            "The company revealed its NFT roadmap for 2025."
        ],
        "brand": [
            "Social media campaigns boosted brand awareness.",
            "Google Trends indicates rising interest in the brand.",
            "The firm was ranked #1 in brand recognition surveys.",
            "Brand loyalty improved with consistent customer engagement.",
            "Influencer partnerships expanded the brand’s visibility."
        ]
    }

    records = []
    for _ in range(n):
        firm = random.choice(firm_names)
        # 随机选择 0~5 个月后的某一月
        month_offset = 2
        date_obj = base_month + relativedelta(months=month_offset)
        month_str = date_obj.strftime("%Y-%m")  # 只保留到月份

        dim = random.choice(list(dimension_templates.keys()))
        text = random.choice(dimension_templates[dim])

        records.append({
            "firm_id": firm,
            "date": month_str,
            "text": text
        })

    return records


if __name__ == "__main__":
    records = generate_test_records(n=100)

    # 保存为 CSV（格式为 firm_id, date (month), text）
    df = pd.DataFrame(records)
    csv_path = "input/example.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 测试数据生成完毕，共 {len(df)} 条，输出至 {csv_path}")
