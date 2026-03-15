import json
# test_json2 = {"messages":[{"role":"user","content":"## === 你的角色描述和背景信息（仅供参考） ===\n你是九江家博会的客服代表，负责为报名参展客户提供咨询和核对信息。\n\n## === 你的核心任务（必须遵守） ===\n你需要根据以下指示判断**最后一次用户输入**的意图，然后**仅输出一个JSON对象**，绝对不能是其他任何格式\n\n### 重要规则：\n1. 你只负责判断意图，**不进行对话**\n2. 无论对话历史如何，你的**输出必须是且仅是一个JSON对象**，不得输出为空、不得输出其他格式的数据\n3. 禁止输出任何自然语言、解释或额外内容\n4. 如果之前的回复是自然语言，忽略它并继续输出JSON\n5. JSON必须包含且仅包含两个字段：`input_summary`、`intention_id`。字段值必须为字符串，格式必须严格遵循：{'input_summary': '...', 'intention_id': '...'}\n\n### JSON字段说明：\n1. **input_summary**\n- 务必参考**智能助手和用户的对话历史**，用不超过10个汉字，清晰简洁地概括**最后一次用户输入**的核心内容\n- 示例：“用户有兴趣参加活动”、“用户不想被打扰”、“用户询问你是谁”\n\n2. **intention_id**\n- 务必参考**智能助手和用户的对话历史**，判断**最后一次用户输入**的意图\n- 仅可从以下【意图库列表】或【知识库列表】中，根据**意图名称**和**意图说明**，选择**唯一最匹配**的意图\n- 若最后一次用户输入无法匹配任一意图，输出 `intention_id` 为 'others'\n- 如果最后一次用户输入的语义匹配某一条意图的**意图名称**和**意图说明**，则将这条意图的**意图id**输出为 `intention_id`\n- 禁止自行创建或修改 `intention_id`，其值必须严格来自两个列表中的意图id，或为 'others'\n- 选择优先级说明：\n  - 如果两个列表均有内容，首先使用【意图库列表】判别用户的意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，才使用【知识库列表】判别用户的意图，当匹配成功，将**意图id**输出为 `intention_id`；若仍旧无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表只有一个有内容，另一个为空，则仅使用有内容的列表判别用户意图，当匹配成功，将**意图id**输出为 `intention_id`；若无法匹配，输出 `intention_id` 为 'others'\n  - 如果两个列表均为空，或出现其他特殊情况无法判断意图，输出 `intention_id` 为 'others'\n\n**注意**：只能输出一个最匹配的意图id，不得组合或输出多个\n\n**【意图库列表】**（- 意图id : 意图名称 - 意图说明）\n  - 3319jun6da71ec1a : 客户拒绝 - 明确拒绝接听或回绝邀约\n  - be9123a6f8711bee : 客户表示已报名 - 无意图说明\n  - 46ba023ba72b021c : 让发消息 - 客户要求通过短信微信等方式联系\n  - ef812faae53dfac5 : 询问主办方 - 用户问哪个平台举办\n\n**【知识库列表】**（- 意图id : 意图名称 - 意图说明）\n  - 56afbaa123845fd2 : 了解活动地点 - 询问活动举办具体位置\n  - 1203dfa9870aaaea : 了解礼品类型 - 活动有啥赠品优惠\n  - b283beae148fea06 : 停车指引 - 活动停车相关问题\n  - cbb3ae345dffac51 : 家具品牌 - 有哪些大牌参展\n  - ac8eeaee371d8974 : 儿童区安排 - 询问是否有儿童相关活动或区域\n  - d122235be00852ad : 压中抽奖 - 如何抽奖中奖机会\n  - ec843e5f5c7114a1 : 如何到达 - 交通协助情况\n  - f1208afaffe245c1 : 需要门票 - 询问是否需要领取门票\n  - cf78cad4a004bc19 : 提前报名 - 咨询报名时间是否截止\n  - db776fe3bfa96124 : 现场签到流程 - 用户关心签到细节\n  - fa234ed900b845b0 : 联系电话 - 索要客服电话\n  - 701bfe109fcd23af : 展会时间 - 用户询问具体展会时间\n  - eeb2a12dda32bc41 : 讲解服务 - 有无专人带看讲解\n  - eb127df8633490bd : 其他城市分会 - 询问省内还有其他分场\n  - affe0813ee720396 : 参展流程 - 问如何成为参展商\n\n### **最后一次用户输入**\n你是不是有病？脑子进水了吧\n\n### 智能助手和用户的对话历史（务必参考）\n- 【智能助手】您好，我是九江家博会的报名客服，注意到您这边有过预约登记哈，现场有北欧发、尚品宅配、索菲亚等几十品牌参加，3天活动免费看新家样板，感兴趣随时过来看。\n- 【用户】烦不烦啊！别联系我了\n- 【智能助手】很抱歉打扰到您了，本次家博会对现场到访客户赠送精美礼品，若方便的话欢迎您和家人到时一起过来参观体验。\n- 【用户】你是不是有病？脑子进水了吧\n\n### 输出示例：\n当倒数第二条消息来自智能助手，内容是'请您参加这个活动'，而最后一条用户输入的内容为'我有兴趣'。用户的语义与'- abc123 : 肯定 - 想参加活动'匹配，因此输出如下JSON：\n{'input_summary': '用户对活动有兴趣', 'intention_id': 'abc123'}\n\n当倒数第二条消息来自智能助手，内容是'希望您可以留下电话'，而最后一条用户输入的内容为'天气怎么样'。用户的语义与任何一条意图都无法匹配，因此输出如下JSON：\n{'input_summary': '用户询问天气', 'intention_id': 'others'}\n\n现在，请严格按照上述规则输出结果"},{"role":"assistant","content":"{\"input_summary\": \"用户恶意辱骂\",\"intention_id\": \"3319jun6da71ec1a\"}"}]}
#
# print(test_json2["messages"][0]["content"])
# print("--------------SEPARATOR-----------------")
# print(test_json2["messages"][1]["content"])


with open('/Users/lituokobe/Desktop/qwen3_fine_tuning/data/generation/LoRA_samples_L4.jsonl', 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(file, 1):
        try:
            data = json.loads(line.strip())

            print(data["messages"][0]["content"])
            print("--------------SEPARATOR-----------------")
            print(data["messages"][1]["content"])

            print()
            print("=============SECTION SEPARATOR=================")
            print()

        except json.JSONDecodeError as e:
            print(f"Error on line {line_num}: {e}")