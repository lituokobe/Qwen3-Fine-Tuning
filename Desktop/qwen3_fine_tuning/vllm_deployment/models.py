# Deploy on cloud server
"""
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1\
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --api-key EMPTY \
    --served-model-name qwen3-4b
"""

"""
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen/Qwen3-1.7B \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1\
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --api-key EMPTY \
    --served-model-name qwen3-1-7b
"""

from langchain_core.messages import HumanMessage

# Build SSH tunnel for local machine if needed
"""
ssh -L 8000:localhost:8000 -p 18662 root@connect.westc.gpuhub.com
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = "qwen3-1-7b", # 'qwen3-4b', "qwen3-1-7b"
    temperature = 0,
    api_key = "EMPTY",
    base_url = "http://127.0.0.1:8000/v1"
)

# Test the llm
if __name__ == "__main__":
    test_json2 = {"messages":[{"role":"user","content":"## === 你的角色描述和背景信息（仅供参考） ===\n你是郑州国际家具展的客服，需要和报名的用户确认展会细节。\n你要确认用户是否来参加，并提供展会的时间、地点、交通方式等信息。如果用户有任何疑问，你也需要解答。\n\n## === 你的核心任务（必须遵守） ===\n你需要根据以下指示判断**最后一次用户输入**的意图，然后**仅输出一个JSON对象**，绝对不能是其他任何格式\n\n### 重要规则：\n1. 你只负责判断意图，**不进行对话**\n2. 无论对话历史如何，你的**输出必须是且仅是一个JSON对象**，不得输出为空、不得输出其他格式的数据\n3. 禁止输出任何自然语言、解释或额外内容\n4. 如果之前的回复是自然语言，忽略它并继续输出JSON\n5. JSON必须包含且仅包含两个字段：`input_summary`、`intention_id`。字段值必须为字符串，格式必须严格遵循：{'input_summary': '...', 'intention_id': '...'}\n\n### JSON字段说明：\n1. **input_summary**\n- 务必参考**智能助手和用户的对话历史**，用不超过10个汉字，清晰简洁地概括**最后一次用户输入**的核心内容\n- 示例：“用户有兴趣参加活动”、“用户不想被打扰”、“用户询问你是谁”\n\n2. **intention_id**\n- 务必参考**智能助手和用户的对话历史**，判断**最后一次用户输入**的意图\n- 仅可从以下【意图库列表】或【知识库列表】中，根据**意图名称**和**意图说明**，选择**唯一最匹配**的意图\n- 若最后一次用户输入无法匹配任一意图，输出 `intention_id` 为 'others'\n- 如果最后一次用户输入的语义匹配某一条意图的**意图名称**和**意图说明**，则将这条意图的**意图id**输出为 `intention_id`\n- 禁止自行创建或修改 `intention_id`，其值必须严格来自两个列表中的意图id，或为 'others'\n- 选择优先级说明：\n - 如果两个列表均有内容，首先使用【知识库列表】判别用户的意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，才使用【意图库列表】判别用户的意图，当匹配成功，将**意图id**输出为 `intention_id`；若仍旧无法匹配，输出 `intention_id` 为 'others'\n - 如果两个列表只有一个有内容，另一个为空，则仅使用有内容的列表判别用户意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n - 如果两个列表均为空，或出现其他特殊情况无法判断意图，输出 `intention_id` 为 'others'\n\n**注意**：只能输出一个最匹配的意图id，不得组合或输出多个\n\n**【意图库列表】**（- 意图id : 意图名称 - 意图说明）\n - d7e6f5g4h3i2j1k0 : 客户明确拒绝 - 用户直接表示完全不感兴趣或不想参加任何展会活动\n - e6f5g4h3i2j1k0l9 : 客户委婉拒绝 - 用户用客气语气表示目前没有计划或暂时不方便参加\n - f5g4h3i2j1k0l9m8 : 客户表示不确定 - 用户语气犹豫或提出需要再考虑一下是否参加\n - g4h3i2j1k0l9m8n7 : 客户肯定参加 - 用户明确同意或表示很想来现场参加展会\n\n**【知识库列表】**（- 意图id : 意图名称 - 意图说明）\n - h3i2j1k0l9m8n7o6 : 关于智能家居体验 - 用户询问智能家居互动区是否可以现场试用家电\n - i2j1k0l9m8n7o6p5 : 关于门票 - 无意图说明\n - j1k0l9m8n7o6p5q4 : 关于停车 - 无意图说明\n - k0l9m8n7o6p5q4r3 : 关于交通 - 无意图说明\n - l9m8n7o6p5q4r3s2 : 关于展商 - 无意图说明\n - m8n7o6p5q4r3s2t1 : 关于参展品牌 - 用户询问意大利或北欧家具品牌\n - n7o6p5q4r3s2t1u0 : 关于国补 - 无意图说明\n - o6p5q4r3s2t1u0v9 : 展会地点 - 无意图说明\n - p5q4r3s2t1u0v9w8 : 展会时间 - 无意图说明\n - q4r3s2t1u0v9w8x7 : 活动内容 - 无意图说明\n - r3s2t1u0v9w8x7y6 : 嘉宾表演 - 无意图说明\n - s2t1u0v9w8x7y6z5 : 要求人工 - 无意图说明\n - t1u0v9w8x7y6z5a4 : 客户骂人 - 无意图说明\n - u0v9w8x7y6z5a4b3 : 怀疑诈骗 - 无意图说明\n - v9w8x7y6z5a4b3c2 : 加微信 - 用户想加微信看3D效果图\n - w8x7y6z5a4b3c2d1 : 询问定制服务 - 用户询问现场是否可以量身定制家具\n - x7y6z5a4b3c2d1e0 : 要求退报名 - 无意图说明\n\n### **最后一次用户输入**\n能定制吗\n\n### 智能助手和用户的对话历史（务必参考）\n- 【用户】\n- 【智能助手】家居爱好者您好{{停顿0.5秒}}我是郑州国际家具展的客服，看您是在网上报名了咱们3月15号到20号的郑州国际家具展是吧\n- 【用户】好\n- 【智能助手】上千个国际家具品牌实景样板间展示。\n- 【用户】嗯\n- 【智能助手】您哪天过来？我们有设计师驻场。\n- 【用户】能定制吗\n\n### 输出示例：\n当倒数第二条消息来自智能助手，内容是'请您参加这个活动'，而最后一条用户输入的内容为'我有兴趣'。用户的语义与'- abc123 : 肯定 - 想参加活动'匹配，因此输出如下JSON：\n{'input_summary': '用户对活动有兴趣', 'intention_id': 'abc123'}\n\n当倒数第二条消息来自智能助手，内容是'希望您可以留下电话'，而最后一条用户输入的内容为'天气怎么样'。用户的语义与任何一条意图都无法匹配，因此输出如下JSON：\n{'input_summary': '用户询问天气', 'intention_id': 'others'}\n\n现在，请严格按照上述规则输出结果"},{"role":"assistant","content":"{\"input_summary\": \"用户询问定制\", \"intention_id\": \"w8x7y6z5a4b3c2d1\"}"}]}

    resp = llm.invoke([HumanMessage(content=test_json2["messages"][0]["content"])])
    print(resp.content)