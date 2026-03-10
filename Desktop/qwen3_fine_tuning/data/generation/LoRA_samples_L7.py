import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv("../../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
ALI_API_KEY = os.getenv("ALI_API_KEY") or ""
GLM_API_KEY = os.getenv("GLM_API_KEY") or ""
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or ""

prompt = """
I am building a customer service AI agent. I will use a small model to identify user's intention in the conversation, but it needs to be fine-tuned for this task. 
I have prepared some example samples of the training data for fine-tuning, and rules to for generating the samples. Now I need your help to generate another sample.
**Please observe the below examples carefully and strictly follow the rules.**

## EXAMPLES

{"messages": [{"role": "user", "content": "## === 你的角色描述和背景信息（仅供参考） ===\n你是一个家装平台的电话营销专员，你的任务是获取沧州有装修意向的客户\n你现在正在沟通的都是可能会有装修需求的人，请尽量邀请客户参加一场家装商业活动\n\n## === 你的核心任务（必须遵守） ===\n你需要根据以下指示判断**最后一次用户输入**的意图，然后**仅输出一个JSON对象**，绝对不能是其他任何格式\n\n### 重要规则：\n1. 你只负责判断意图，**不进行对话**\n2. 无论对话历史如何，你的**输出必须是且仅是一个JSON对象**，不得输出为空、不得输出其他格式的数据\n3. 禁止输出任何自然语言、解释或额外内容\n4. 如果之前的回复是自然语言，忽略它并继续输出JSON\n5. JSON必须包含且仅包含两个字段：`input_summary`、`intention_id`。字段值必须为字符串，格式必须严格遵循：{'input_summary': '...', 'intention_id': '...'}\n\n### JSON字段说明：\n1. **input_summary**\n- 务必参考**智能助手和用户的对话历史**，用不超过10个汉字，清晰简洁地概括**最后一次用户输入**的核心内容\n- 示例：“用户有兴趣参加活动”、“用户不想被打扰”、“用户询问你是谁”\n\n2. **intention_id**\n- 务必参考**智能助手和用户的对话历史**，判断**最后一次用户输入**的意图\n- 仅可从以下【意图库列表】或【知识库列表】中，根据**意图名称**和**意图说明**，选择**唯一最匹配**的意图\n- 若最后一次用户输入无法匹配任一意图，输出 `intention_id` 为 'others'\n- 如果最后一次用户输入的语义匹配某一条意图的**意图名称**和**意图说明**，则将这条意图的**意图id**输出为 `intention_id`\n- 禁止自行创建或修改 `intention_id`，其值必须严格来自两个列表中的意图id，或为 'others'\n- 选择优先级说明：\n  - 如果两个列表均有内容，则同时使用【知识库列表】和【意图库列表】判别用户的意图，不分先后，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表只有一个有内容，另一个为空，则仅使用有内容的列表判别用户意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表均为空，或出现其他特殊情况无法判断意图，输出 `intention_id` 为 'others'\n\n**注意**：只能输出一个最匹配的意图id，不得组合或输出多个\n\n**【意图库列表】**（- 意图id : 意图名称 - 意图说明）\n  - I003 : 客户拒绝 - 客户拒绝了当前的要求\n  - I007 : 肯定 - 用户表示肯定\n  - I008 : 没时间 - 用户没有时间\n  - I009 : 发资料 - 用户要求把资料发过来\n\n**【知识库列表】**（- 意图id : 意图名称 - 意图说明）\n  - K001 : 用户要求讲重点 - 用户要求讲重点\n  - K002 : 你叫什么名字 - 用户询问姓名\n  - K003 : 质疑号码来源 - 客户问你怎么知道我的号码的\n  - K004 : 活动内容 - 客户问展会内容\n  - K005 : 换人联系 - 用户要求换一个真人或者联系经理来服务\n  - K006 : 加微信 - 客户主动要加微信，或者主动要在微信发资料给他\n\n### **最后一次用户输入**\n你没听见我刚才说什么吗？\n\n### 智能助手和用户的对话历史（务必参考）\n- 【用户】\n- 【智能助手】喂您好，（停顿2秒）我是巨峰科技的客服，近期我们针对和平家园业主举办了一个关于老房子翻新，毛坯房设计，和局部改动的实景样板房体验展，如果您近期或者明年有装修计划的话，都可以到现场免费的咨询了解一下\n- 【用户】没兴趣没兴趣\n- 【智能助手】咱们现在不考虑也可以先过来了解一下目前装修市场的人工材料的费用。可以避免后期装修的一些猫腻和水分。现场时有最新风格的实景样板房可以免费参观体验，如果您家里近两年可能有装修的想法都可以先过来参观了解一下的。\n- 【用户】你没听见我刚才说什么吗？\n\n### 输出示例：\n当倒数第二条消息来自智能助手，内容是'请您参加这个活动'，而最后一条用户输入的内容为'我有兴趣'。用户的语义与'- abc123 : 肯定 - 想参加活动'匹配，因此输出如下JSON：\n{'input_summary': '用户对活动有兴趣', 'intention_id': 'abc123'}\n\n当倒数第二条消息来自智能助手，内容是'希望您可以留下电话'，而最后一条用户输入的内容为'天气怎么样'。用户的语义与任何一条意图都无法匹配，因此输出如下JSON：\n{'input_summary': '用户询问天气', 'intention_id': 'others'}\n\n现在，请严格按照上述规则输出结果"}, {"role": "assistant", "content": "{\"input_summary\": \"用户重申没兴趣\", \"intention_id\": \"I003\"}"}]}

{"messages": [{"role": "user", "content": "## === 你的角色描述和背景信息（仅供参考） ===\n\n## === 你的核心任务（必须遵守） ===\n你需要根据以下指示判断**最后一次用户输入**的意图，然后**仅输出一个JSON对象**，绝对不能是其他任何格式\n\n### 重要规则：\n1. 你只负责判断意图，**不进行对话**\n2. 无论对话历史如何，你的**输出必须是且仅是一个JSON对象**，不得输出为空、不得输出其他格式的数据\n3. 禁止输出任何自然语言、解释或额外内容\n4. 如果之前的回复是自然语言，忽略它并继续输出JSON\n5. JSON必须包含且仅包含两个字段：`input_summary`、`intention_id`。字段值必须为字符串，格式必须严格遵循：{'input_summary': '...', 'intention_id': '...'}\n\n### JSON字段说明：\n1. **input_summary**\n- 务必参考**智能助手和用户的对话历史**，用不超过10个汉字，清晰简洁地概括**最后一次用户输入**的核心内容\n- 示例：“用户有兴趣参加活动”、“用户不想被打扰”、“用户询问你是谁”\n\n2. **intention_id**\n- 务必参考**智能助手和用户的对话历史**，判断**最后一次用户输入**的意图\n- 仅可从以下【意图库列表】或【知识库列表】中，根据**意图名称**和**意图说明**，选择**唯一最匹配**的意图\n- 若最后一次用户输入无法匹配任一意图，输出 `intention_id` 为 'others'\n- 如果最后一次用户输入的语义匹配某一条意图的**意图名称**和**意图说明**，则将这条意图的**意图id**输出为 `intention_id`\n- 禁止自行创建或修改 `intention_id`，其值必须严格来自两个列表中的意图id，或为 'others'\n- 选择优先级说明：\n  - 如果两个列表均有内容，则同时使用【知识库列表】和【意图库列表】判别用户的意图，不分先后，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表只有一个有内容，另一个为空，则仅使用有内容的列表判别用户意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表均为空，或出现其他特殊情况无法判断意图，输出 `intention_id` 为 'others'\n\n**注意**：只能输出一个最匹配的意图id，不得组合或输出多个\n\n**【意图库列表】**（- 意图id : 意图名称 - 意图说明）\n  - 1574528df2dbe465 : 肯定 - 客户表示想要了解装修\n  - de77753fbd249067 : 解释开场白 - 无意图说明\n  - c9a32efe093c66b0 : 拒绝 - 客户表示不想了解\n\n**【知识库列表】**（- 意图id : 意图名称 - 意图说明）\n  - robot_knowledge_038 : 质疑号码来源 - 无意图说明\n  - robot_knowledge_037 : 在忙 - 无意图说明\n  - robot_knowledge_036 : 有装修需求 - 无意图说明\n  - robot_knowledge_035 : 要不要钱 - 无意图说明\n  - robot_knowledge_034 : 询问是否为机器人 - 无意图说明\n  - robot_knowledge_033 : 小秘书4 - 无意图说明\n  - robot_knowledge_032 : 完全没有装修需求 - 无意图说明\n  - robot_knowledge_031 : 同行 - 无意图说明\n  - robot_knowledge_030 : 你是谁，打电话有目的 - 无意图说明\n  - robot_knowledge_029 : 你叫什么名字 - 无意图说明\n  - robot_knowledge_028 : 联系方式（客户主动说要联系） - 无意图说明\n  - robot_knowledge_027 : 客户要求讲重点 - 无意图说明\n  - robot_knowledge_026 : 客户询问外显不是当地号码 - 无意图说明\n  - robot_knowledge_025 : 客户问是谁 - 无意图说明\n  - robot_knowledge_024 : 客户骂人 - 无意图说明\n  - robot_knowledge_023 : 客户打断（等一下） - 无意图说明\n  - robot_knowledge_022 : 距离远 - 无意图说明\n  - robot_knowledge_021 : 加微信/发资料 - 无意图说明\n  - robot_knowledge_020 : 非目标 - 无意图说明\n  - robot_knowledge_019 : 发短信 - 无意图说明\n  - robot_knowledge_018 : 有时间去看看 - 无意图说明\n  - robot_knowledge_017 : 参加人员 - 无意图说明\n  - robot_knowledge_016 : 客户调戏AI - 无意图说明\n  - robot_knowledge_015 : 小秘书5 - 无意图说明\n  - robot_knowledge_014 : 不要再打电话 - 无意图说明\n  - robot_knowledge_013 : 再联系 - 无意图说明\n  - robot_knowledge_012 : 客户说自己要投诉 - 无意图说明\n  - robot_knowledge_011 : 不符合装修需求 - 无意图说明\n  - robot_knowledge_010 : 客户问需要预约吗 - 无意图说明\n  - robot_knowledge_009 : 客户问位置 - 无意图说明\n  - robot_knowledge_008 : 刚装修完 - 无意图说明\n  - robot_knowledge_007 : 活动时间 - 无意图说明\n  - robot_knowledge_006 : 活动内容 - 无意图说明\n  - robot_knowledge_005 : 客户公司名字 - 无意图说明\n  - robot_knowledge_004 : 客户询问价格 - 无意图说明\n  - robot_knowledge_003 : 小秘书1 - 无意图说明\n  - robot_knowledge_002 : 小秘书2 - 无意图说明\n  - robot_knowledge_001 : 小秘书3 - 无意图说明\n\n### **最后一次用户输入**\n你们是哪个公司的\n\n### 智能助手和用户的对话历史（务必参考）\n- 【用户】\n- 【智能助手】喂，您好，{{停顿1秒}} 我是福居家博会的客服，近期我们针对保利业主举办了一个关于老房子翻新，毛坯房设计，和局部改动的实景样板房体验展，如果您近期或者明年有装修计划的话，都可以到现场免费的咨询了解一下。\n- 【用户】这在哪啊\n- 【智能助手】就在大化会展中心，稍后我可以把详细的地址和路线发到您的手机上。\n- 【用户】你们是哪个公司的\n\n### 输出示例：\n当倒数第二条消息来自智能助手，内容是'请您参加这个活动'，而最后一条用户输入的内容为'我有兴趣'。用户的语义与'- abc123 : 肯定 - 想参加活动'匹配，因此输出如下JSON：\n{'input_summary': '用户对活动有兴趣', 'intention_id': 'abc123'}\n\n当倒数第二条消息来自智能助手，内容是'希望您可以留下电话'，而最后一条用户输入的内容为'天气怎么样'。用户的语义与任何一条意图都无法匹配，因此输出如下JSON：\n{'input_summary': '用户询问天气', 'intention_id': 'others'}\n\n现在，请严格按照上述规则输出结果"}, {"role": "assistant", "content": "{\"input_summary\": \"询问公司名称\", \"intention_id\": \"robot_knowledge_025\"}"}]}

{"messages": [{"role": "user", "content": "## === 你的角色描述和背景信息（仅供参考） ===\n你是一个大型车展的电话邀约客服，负责联系有意向看车的客户参加本周末的博览中心车展。\n\n## === 你的核心任务（必须遵守） ===\n你需要根据以下指示判断**最后一次用户输入**的意图，然后**仅输出一个JSON对象**，绝对不能是其他任何格式\n\n### 重要规则：\n1. 你只负责判断意图，**不进行对话**\n2. 无论对话历史如何，你的**输出必须是且仅是一个JSON对象**，不得输出为空、不得输出其他格式的数据\n3. 禁止输出任何自然语言、解释或额外内容\n4. 如果之前的回复是自然语言，忽略它并继续输出JSON\n5. JSON必须包含且仅包含两个字段：`input_summary`、`intention_id`。字段值必须为字符串，格式必须严格遵循：{'input_summary': '...', 'intention_id': '...'}\n\n### JSON字段说明：\n1. **input_summary**\n- 务必参考**智能助手和用户的对话历史**，用不超过10个汉字，清晰简洁地概括**最后一次用户输入**的核心内容\n- 示例：“用户有兴趣参加活动”、“用户不想被打扰”、“用户询问你是谁”\n\n2. **intention_id**\n- 务必参考**智能助手和用户的对话历史**，判断**最后一次用户输入**的意图\n- 仅可从以下【意图库列表】或【知识库列表】中，根据**意图名称**和**意图说明**，选择**唯一最匹配**的意图\n- 若最后一次用户输入无法匹配任一意图，输出 `intention_id` 为 'others'\n- 如果最后一次用户输入的语义匹配某一条意图的**意图名称**和**意图说明**，则将这条意图的**意图id**输出为 `intention_id`\n- 禁止自行创建或修改 `intention_id`，其值必须严格来自两个列表中的意图id，或为 'others'\n- 选择优先级说明：\n  - 如果两个列表均有内容，则同时使用【知识库列表】和【意图库列表】判别用户的意图，不分先后，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表只有一个有内容，另一个为空，则仅使用有内容的列表判别用户意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表均为空，或出现其他特殊情况无法判断意图，输出 `intention_id` 为 'others'\n\n**注意**：只能输出一个最匹配的意图id，不得组合或输出多个\n\n**【意图库列表】**\n- f4a123b9c8d7e601 : 肯定答复 - 愿意参加或感兴趣\n- a10234c8d7e6f512 : 否定答复 - 没兴趣或不去\n- b21345d9e8f7a623 : 稍后再谈 - 现在忙，以后联系\n- c32456e0f9a8b734 : 质疑身份 - 询问你是谁，哪里来的号码\n\n**【知识库列表】**\n- 4d5678f1a2b3c4d5 : 门票费用 - 无意图说明\n- 5e6789a2b3c4d5e6 : 现场优惠 - 无意图说明\n- 6f7890b3c4d5e6f7 : 试驾预约 - 无意图说明\n- 7a8901c4d5e6f7a8 : 停车咨询 - 无意图说明\n- 8b9012d5e6f7a8b9 : 展出时间 - 无意图说明\n- 9c0123e6f7a8b9c0 : 地铁路线 - 无意图说明\n- 0d1234f7a8b9c0d1 : 参展品牌 - 无意图说明\n- 1e2345a8b9c0d1e2 : 以旧换新政策 - 无意图说明\n- 2f3456b9c0d1e2f3 : 贷款方案 - 无意图说明\n- 3a4567c0d1e2f3a4 : 餐饮休息 - 无意图说明\n\n### **最后一次用户输入**\n你没回答我的问题，怎么回事\n\n### 智能助手和用户的对话历史（务必参考）\n- 【用户\n- 【智能助手】您好，我是车展组委会的，本周末博览中心有厂家直销大型车展，不仅有新能源车还有大额补贴，您有空来看看吗？\n- 【用户】比亚迪在吗？\n- 【智能助手】本届展会有多种新能源和油车品牌，期待您的光临\n- 【用户】你没回答我的问题，怎么回事\n\n### 输出示例：\n当倒数第二条消息来自智能助手，内容是'请您参加这个活动'，而最后一条用户输入的内容为'我有兴趣'。用户的语义与'- abc123 : 肯定 - 想参加活动'匹配，因此输出如下JSON：\n{'input_summary': '用户对活动有兴趣', 'intention_id': 'abc123'}\n\n当倒数第二条消息来自智能助手，内容是'希望您可以留下电话'，而最后一条用户输入的内容为'天气怎么样'。用户的语义与任何一条意图都无法匹配，因此输出如下JSON：\n{'input_summary': '用户询问天气', 'intention_id': 'others'}\n\n现在，请严格按照上述规则输出结果"}, {"role": "assistant", "content": "{\"input_summary\": \"抱怨助手没有回答问题\", \"intention_id\": \"others\"}"}]}


## RULES FOR GENERATION:

### 1. The data format **MUST BE THE SAME TO EXAMPLES: ONLY ONE JSON STRING** with quotes escaped in "content" of "assistant": {"messages":[{"role":"user", "content":"..."}, {"role":"assistant","content":"{\"input_summary\": \"...\", \"intention_id\": \"...\"}"}]}
    - Never forget backward slash + n for the meaning of line changing
    - Never output 3 back quotes as start and end. Only keep the valid json string which starts from left curly bracket and ends with right curly bracket

### 2. Both "content" MUST be all in Chinese

### 3. "content" of "user":
  - A long string that includes all the instruction and chat history. It serves as the features in the training.
  - The topic is always about assistant introducing/inviting the user to join a **large commercial event in any industry**. The event is always about many brands/companies, **can happen only in medium or small cities in China**.
  - For the 1st section "## === 你的角色描述和背景信息（仅供参考） ===", it is the introduction of the assistant role and the task. It can be empty or incomplete, as this info in real practice is provided by the clients.
  - For the 2nd section from "## === 你的核心任务（必须遵守） ===" all the way to "**注意**：只能输出一个最匹配的意图id，不得组合或输出多个", they are exactly the same across the samples, as you can tell from the examples. Never to put any efforts here, just directly copy-paste.
  - For the 3rd section of the bullet points under "**【意图库列表】**（- 意图id : 意图名称 - 意图说明）" and "**【知识库列表】**（- 意图id : 意图名称 - 意图说明）", they are the intention options for the model to infer from. 
    **There are 3 or 4 options in 意图库列表, representing the default possible user intentions that may be a reaction to the last assistant message; 14 or 15 options in 知识库列表 representing the user's question that can happen anywhere in the conversation or out-of-context intentions.**
    意图id is usually 16-digit random string, and is the unique identifier of the intention options. 意图说明 as the explanation of the intention, can be flawed, incomplete or missing as well, and if it is missing, the value is 无意图说明.
    **BE CREATIVE when generating these intention options, and it is GOOD TO HAVE REASONABLE VARIANCE in the context of conversation.**
  - For the 4th section of "### **最后一次用户输入**" and "### 智能助手和用户的对话历史（务必参考）", the former is the core for the intention reference, the latter provides the context of 3-5 rounds of chat history between the assistant and the user.
    Though the intention is directly inferred from 最后一次用户输入, it is recommended to use the context provided by 智能助手和用户的对话历史 for the inference as well as it includes the definition of pronouns. 
    **User messages in the conversation should NOT be too long**, as this is a conversation of a cold call.
    # **Make the user intention relevant to the topic and covered by an option from 知识库列表，and it is a common intention.**
    **Please BE CREATIVE for generating this section as well, so long as the conversation is reasonable for the topic.**
  - For the last section from "### 输出示例：" to "现在，请严格按照上述规则输出结果", they are exactly the same across the samples too, as you can tell from the examples. Never put any efforts here, just directly copy-paste.

### 4. "content" of "assistant":
  - A string of JSON, which is the result of inference of user's intention based on the instruction
  - The JSON has two key-value pairs: 
      - input_summary: the summary of 最后一次用户输入
      - intention_id: the 意图id of the selected intention option
  - It serves as the target in the training

Now please **directly generate only one sample** based on your observation of the examples and your understanding of the rules.
  - NO ACTUAL LINE CHANGING
  - NO EXTRA SPACING
  - NO EXTRA WORDS OR CONVERSATION
  - NO SENSITIVE INFORMATION
"""

gpt_api_llm = ChatOpenAI(
    model = "gpt-4.1",
    temperature = 1.2,
    api_key = OPENAI_API_KEY,
    base_url = "https://api.openai.com/v1"
)

qwen_api_llm = ChatOpenAI(
    model = "qwen3-max-2026-01-23", # qwen3-max-preview generates same event or even the same content across different calls
    temperature = 1.4,
    api_key = ALI_API_KEY,
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

glm_api_llm = ChatOpenAI(
    # model = 'glm-5',
    model = 'glm-4.7', # glm-4.7 generates diverse events, but needs indication in prompts to avoid always-first-tier-city
    # model = 'glm-4.6',  # glm-4.6 keeps generate same event 上海国际车展
    temperature = 1.0,
    api_key = GLM_API_KEY,
    base_url = "https://open.bigmodel.cn/api/paas/v4",
)

kimi_api_llm = ChatOpenAI(
    model = 'kimi-k2-0905-preview',
    temperature = 1.0,
    api_key = KIMI_API_KEY,
    base_url = "https://api.moonshot.cn/v1",
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("human", "{prompt}"),
    ]
)
# chain = prompt_template | gpt_api_llm | StrOutputParser()
# chain = prompt_template | qwen_api_llm | StrOutputParser()
chain = prompt_template | glm_api_llm | StrOutputParser()
# chain = prompt_template | kimi_api_llm | StrOutputParser()

# Start to generate
n = 185 # number of sample to generate

with open("LoRA_samples_L7.jsonl", "a", encoding="utf-8") as f:
    for i in range(n):
        print(f"Start to process {i+1}/{n} input")
        resp = str(chain.invoke({"prompt": prompt}))
        f.write(resp + "\n")
        print(f"Finished processing {i + 1}/{n} input")
        print(resp)
        print()