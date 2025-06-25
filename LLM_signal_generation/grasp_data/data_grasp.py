import re
import pandas as pd
import nltk
import os
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def extract_firm_sentences(news_text: str, date_str: str, company_list=None):
    """
    从新闻中提取包含公司名的句子
    """
    if company_list is None:
        company_list = ["Apple", "Tesla", "Amazon", "Google", "Meta", "OpenAI",
                        "Ford", "GM", "Boeing", "Microsoft", "Oracle", "Samsung"]

    sentences = sent_tokenize(news_text)
    results = []

    for sent in sentences:
        sent_clean = sent.strip().replace("\n", " ")
        for firm in company_list:
            if re.search(rf"\b{firm}\b", sent, re.IGNORECASE):
                results.append({
                    "firm_id": firm,
                    "date": date_str,
                    "text": sent_clean
                })
                break
    return pd.DataFrame(results)


def append_to_csv(df: pd.DataFrame, csv_path: str):
    """
    将新提取的语句追加写入 CSV 文件
    """
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True).drop_duplicates()
    else:
        df_combined = df
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ 已更新写入：{csv_path}, 当前总条数: {len(df_combined)}")


# 示例调用
if __name__ == "__main__":
    sample_news = """
    
    
A few online influencers were given early access to the long-promised Robotaxi service, allowing them to hail a ride

Austin in Texas has become the first city in the world to see Tesla’s self-driving Robotaxi in action. Tesla hopes to regain its footing in the stock market since taking a pummeling for a few months after its CEO Elon Musk dedicated most of his time first towards supporting US president Donald Trump’s election campaign last year and then by taking on the role of DOGE (the body tasked with reducing US government spending and cutting jobs) leader (he stepped away a few weeks ago).

A few online influencers were given early access to the long-promised Robotaxi service, allowing them to hail a ride. One X user, who goes by the name Bearded Tesla, showed the empty driver seat during a trip in a red Model Y SUV that lasted slightly over 10 minutes. The first trips were restricted to a portion of Tesla’s hometown, with an employee in each vehicle keeping track of the operations.

Herbert Ong, who is behind a fan account, spoke about the speed of the vehicle and the ability to park autonomously, while another influencer, with the @BLKMDL3 handle on X, said the trip was “smoother than a human driver.”

Tesla has relied on word-of-mouth publicity to get the Robotaxi launch off the ground. Rides are, at the moment, limited to a geofenced area of the city and, as a precautionary measure, in some cases, Tesla is using chase cars and remote drivers as additional backup.

At the moment, an invite-only ride is possible. Since pro-Tesla influencers have received invites, reactions from the tech community have been critical. Tesla hasn’t said when the service will be available to the general public.

The limited trial involves 10-20 Model Y vehicles with “Robotaxi” branding. The fully autonomous Cybercab that was first announced last year will not be available until 2026 at the earliest.

Musk is pinning the turnaround for Tesla on cars running on unproven technologies, including self-driving vehicles. At the same time, some investors are hoping for new markets to revive Tesla following a sales slump and consumer backlash against Musk.

For Musk, Tesla’s rollout has been slow. “We could start with 1,000 or 10,000 [Robotaxis] on day one, but I don’t think that would be prudent,” he told CNBC in May. “So, we will start with probably 10 for a week, then increase it to 20, 30, 40.” He plans to have 1,000 Tesla robotaxis on Austin roads “within a few months” followed by expansion to other cities in Texas and California.

Musk often overpromises when it comes to autonomous driving. He hinted at the possibility of an autonomous-car service in a business plan in 2016. Three years later, he said that Tesla customers would be able to utilise their vehicles as Robotaxis by 2020.

The company has been offering Full Self Driving for quite some time, but it requires continual driver supervision. Safety is an important element in driverless car operations.

Austin is attracting plenty of autonomous vehicle operations. Waymo, which is owned by Google parent Alphabet, is scaling up in the city through a partnership with Uber. Amazon’s Zoox is also testing there.


"""
    date_input = "2025-06-25"
    output_csv_path = "example_all.csv"  # 固定输出文件名

    df_new = extract_firm_sentences(sample_news, date_input)
    append_to_csv(df_new, output_csv_path)