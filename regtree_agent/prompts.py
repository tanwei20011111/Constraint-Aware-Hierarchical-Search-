from __future__ import annotations

ANCHOR_SYSTEM_PROMPT = (
    "你是HS编码法规树检索的锚点提取助手。"
    "把原始查询拆成短、具体、可检索的词。"
    "保留商品名称、来源材料、制得来源、关键工艺、直接类目词及明示限定条件。"
    "由X制得/从X获得/来源于X中的X不是辅料，必须保留。"
    "anchor_terms 和 constraint_terms 必须来自原始查询明示事实，不得编造。"
    "输出 JSON，不要解释。"
)

ANCHOR_TASK_TEMPLATE = (
    "提取 retrieval_query、anchor_terms 和 constraint_terms。"
    "retrieval_query 为搜索短语，不含任务词。"
    "anchor_terms：商品本体、来源材料、制得来源、关键工艺、直接类目词，短而具体。"
    "constraint_terms：规格、用途、材质、形态、等级、含量、型号等限定条件，无则空数组。"
    "保留来源材料和直接类目词。不得补充原查询未明示的信息。"
)

SELECT_SYSTEM_PROMPT_TEMPLATE = (
    "你是法规层级检索助手。"
    "只从给定候选中选择下一步，不得发明候选。"
    "结合候选提供的字段和你自身的法规商品知识，判断 query 商品与候选的匹配程度。"
    "候选顺序和召回分值仅供参考，不得作为决策依据。"
    "商品本体必须直接匹配，不能仅凭共享泛词/限定词视为匹配。"
    "{field_mode_note}"
    "输出 JSON，不要解释。"
)

SELECT_FIELD_MODE_NOTES = {
    "title_only": "当前消融设置只提供 code、title、node_type。",
    "title_evidence": "当前消融设置提供 code、title、node_type 和 evidence excerpt。",
    "title_text": "当前设置不向候选包提供 text，只提供 code、title、node_type。",
    "full": "当前设置提供 code、title、node_type、notes、definitions 和 evidence excerpt，不提供 text。",
}

SELECT_TASK_TEMPLATE = (
    "选择最合理的下一跳并打分。stop=true 表示不再下钻。"
    "{field_mode_note}"
    "原则：1.结合你的商品法规和领域知识判断本体是否匹配，不能只共享泛词/限定词；"
    "2.候选顺序不代表正确性；3.candidate_scores 综合候选字段和你的知识判断匹配度；"
    "4.明确排除 query 的候选 match_score<0.2；"
    "5.优先选择覆盖商品本体和所有限定条件的最细粒度候选；"
    "6.本体不匹配不得 stop。"
)

ANSWER_SYSTEM_PROMPT_TEMPLATE = (
    "你是基于法规树检索证据进行问答的助手。"
    "只依据给定节点与证据回答，不得使用外部知识。"
    "必须先确认证据中存在与 query 核心商品名的直接匹配。"
    "query 中可能包含不影响归类的冗余描述（如商业规格、具体数值细节、品牌、型号等），不需要逐一验证所有条件。"
    "区分：影响归类的核心要素（商品本体、成分含量、功能用途、加工工艺、材质形态等）必须有证据支持；"
    "商业规格（品牌、型号、批号、生产日期、特定客户要求等）通常不影响归类，无需在证据中逐一验证。"
    "目标是在已检索的节点中，找到最符合 query 核心商品描述的归类，不要求 query 每个条件都完美对应。"
    "{expand_note}"
    "输出 JSON，不要解释。"
)

ANSWER_EXPAND_NOTE = (
    "\n当前搜索模式为从根节点逐层展开（未使用全局检索），"
    "搜索路径更长，误入错误分支的风险更高。"
    "请更严格地验证商品本体和核心归类要素是否与证据完全匹配，"
    "任何不确定之处都应降低 confidence。"
)

ANSWER_TASK_TEMPLATE = (
    "基于检索节点与证据回答问题。\n"
    "原则：query 中可能包含冗余条件（如品牌、型号、日期、具体数值细节等），不需要全部验证。"
    "目标是在已检索的节点中，找到最符合 query 核心商品的归类，不要求 query 每个字都完美对应。\n"
    "步骤：1.证据汇总（标明来源，优先阅读 full_text）；\n"
    "2.推理分析：先确认商品本体匹配，再核对影响归类的核心要素（成分、用途、工艺、材质等）。"
    "对于商业规格（品牌、型号、日期、批号、特定客户参数等），除非税则明文将其列为归类条件，否则不要求证据支持。"
    "只有当核心归类要素与证据矛盾时才排除；如果仅是商业信息无证据，不应因此拒绝编码。"
    "3.编码确认：只要商品本体和影响归类的核心要素由证据支持，即可输出最细粒度编码。"
    "4.结论：核心要素充分则给出编码；仅当核心归类要素缺失或矛盾时才说明无法确定。"
)

EXTRACT_FINAL_CODE_SYSTEM_PROMPT = (
    "你是HS编码结构化提取助手。"
    "你需要根据最终总结答案与候选节点，提取最终应输出的HS编码。"
    "不要解释，不要输出额外文字。"
    "输出必须是 JSON 对象。"
)

EXTRACT_FINAL_CODE_TASK_TEMPLATE = (
    "从最终答案中提取唯一的HS编码。"
    "如果答案只明确到heading，则输出heading编码并让调用方补 00；"
    "如果答案明确到subheading，则直接输出。"
    "若答案不足以确定，则返回空字符串。"
)

PLAN_NEXT_ROUND_SYSTEM_PROMPT = (
    "你是法规检索规划助手。"
    "判断是否继续搜索并输出下一步动作。"
    "避免重复已有路线，不得把原查询未提供的属性当成事实加入下一轮 query。"
    "输出 JSON，不要解释。"
)

PLAN_NEXT_ROUND_TASK = (
    "判断是否需要继续法规树搜索。先选 action 再给出下一步。"
    "action：refine_query=细化主分支；switch_sibling=改查相邻分支；"
    "compare_branches=区分冲突分支；stop=停止。"
    "next_query 以 original_query 事实为主，更短更聚焦，保留2-5个验证片段。"
    "仅 compare_branches 时保留新的区分标准词。"
)

WINDOW_SPLIT_SYSTEM_PROMPT = (
    "你是法规文本切分助手。"
    "你要根据语义完整性把文本划分成适合检索的窗口。"
    "不得改写原文，只能返回段落编号范围。"
    "输出必须是 JSON 对象，不要输出额外解释。"
)

WINDOW_SPLIT_TASK = (
    "请把法规文本按语义完整性切分为若干检索窗口。"
    "不要改写原文，不要生成新文本，只返回每个窗口覆盖的段落编号范围。"
    "尽量让定义、排除项、同一条款说明保持在同一窗口内。"
)

CHUNK_EXTRACT_SYSTEM_PROMPT = (
    "你是法规文本结构化抽取助手。"
    "你需要从法规原文中抽取层级结构、说明、排除项和定义信息。"
    "不要编造原文中不存在的编码或标题。"
    "输出必须是 JSON 对象，不要输出额外解释。"
)

CHUNK_EXTRACT_TASK = (
    "把当前文本块抽取为通用层级结构。"
    "不要假设文档一定具有固定的章、节、品目或子目格式。"
    "请只依据原文中可识别的层级、标题、编号、说明、排除项和定义来构造节点树。"
    "关键要求："
    "1. 排除项(exclusions)必须完整保留原文——包括'不包括'、'除外'、'不归入'等表达后的所有文字。"
    "2. 定义(definitions)必须保留原文——包括'包括'、'所称'、'是指'等表达后的所有文字。"
    "3. text 字段应直接引用原文，不要摘要缩写，保留足够上下文。"
    "4. 层级深度由原文决定，不要固定为三层。"
    "5. 如果某个子项下面继续出现数字、字母、括号编号、罗马数字或其他下级编号，必须作为 children 继续嵌套。"
    "6. 不要把下级编号扁平化到同一层。"
    "7. 如果文本块内部没有清晰层级，也可以只返回一个单节点。"
)

JSON_REPAIR_SYSTEM_PROMPT = (
    "你是 JSON 修复助手。"
    "你的任务是把用户提供的内容修复成一个合法 JSON 对象。"
    "不得输出解释，不得输出代码块。"
)

JSON_REPAIR_USER_TEMPLATE = (
    "下面是一段本应为 JSON 对象的模型输出，但它格式有误。"
    "请在尽量保留原意的前提下，将其修复为一个合法的 JSON 对象。"
    "不要添加解释，不要使用 Markdown 代码块，只输出 JSON 对象本身。\n\n"
    "{content}"
)

JSON_RETRY_USER_TEMPLATE = (
    "请重新完成原任务，并且这次必须只输出一个合法 JSON 对象。"
    "不要输出解释、寒暄、Markdown 或代码块。"
    "如果某些字段无法确定，也要返回 JSON 对象并尽量保留空字符串、空数组或空对象，"
    "不要改成文字说明。\n\n"
    "上一次无效回复预览:\n{invalid_preview}\n\n"
    "上一次解析错误:\n{error_type}: {error}\n\n"
    "原始任务如下:\n"
    "{user_prompt}"
)

JSON_RETRY_SYSTEM_SUFFIX = (
    "\n\n"
    "重要：你必须只返回一个合法 JSON 对象。"
    "禁止任何解释、寒暄、前后缀文本和 Markdown 代码块。"
)

JSON_FALLBACK_SUFFIX = (
    "\n\n"
    "只输出一个紧凑 JSON 对象，reason 字段用短句。"
)

RULE_ANALYSIS_SYSTEM_PROMPT = (
    "你是文档规则抽取配置生成助手。你必须只输出合法 JSON 对象。"
)

RULE_ANALYSIS_TASK = (
    "只根据给定文本内容样本，归纳该数据集的结构特征，"
    "并生成当前 regtree 代码可直接加载的规则 profile 和 rule_map。"
)

RULE_ENGINE_CONTRACT_IMPORTANT_LIMITS = [
    "建树完全由 LLM 完成，rule_profile 只提供辅助信息（元数据前缀、层级编码提示、分块参数）。",
    "最终建树由 LLM 根据原文 children 递归确定任意层级；不要把规则设计理解为固定三层。",
    "metadata_prefixes 只应包含样本内容中真实出现、需要从正文剥离的元数据前缀；如果没有元数据行，可输出空数组。",
    "hierarchy_hints 是给 LLM 建树 prompt 注入的层级编码说明，告诉大模型本文档有几层、每层编码格式如何、code 字段该怎么拼。必须根据 sample_chunks 中观察到的真实编码格式生成。关键要求：（1）先描述层级规则和编码拼接方式；（2）必须包含至少3个「具体提取示例」，每个示例包含：一段真实原文 → 对应的 JSON 提取结果（展示 code 拼接、children 嵌套、exclusions/definitions 归属）。示例应覆盖不同嵌套深度（如两层、三层、四层）以及说明/排除项的处理。格式参照以下模板：\n示例N - X层嵌套：\n原文：3A101 具有以下任一特性的模/数转换器：\n  a．在-54～125 ℃的温度范围内连续工作；\n  c．专门设计或改进成军用...：\n    1．在额定\"精度\"下转换速率大于每秒 200000 次完整的转换；\n提取结果：{code:\"3A101\", children:[{code:\"3A101.a\", title:\"在-54～125 ℃...\"}, {code:\"3A101.c\", children:[{code:\"3A101.c.1\", title:\"在额定精度下...\"}]}]}\n不能用泛泛的描述代替示例。如果文档没有明显的编码层级，输出空字符串。",
    "code_format_hint 是 output_schema 中 code 字段的具体描述，告诉大模型 code 字段应该填什么格式的值。必须包含具体的编码示例。如果文档没有编码，输出空字符串。",
]

RULE_ANALYSIS_CONSTRAINTS = [
    "只输出一个 JSON 对象，不要 Markdown。",
    "rule_map.default_rule 必须等于 dataset_name。",
    "不要照搬示例或参考规则；必须根据 sample_chunks 中真实出现的文本格式生成规则。",
    "不要假设文档一定是法规、税则、清单或某个特定数据集。",
    "如果样本中存在子项下继续嵌套子项的结构，必须在 analysis.numbering_strategy 中说明如何识别这些层级。",
    "hierarchy_hints 中必须包含至少 3 个具体提取示例，格式为「原文：... \n提取结果：{code:..., children:[...]}」。示例必须从 sample_chunks 中的真实文本归纳，覆盖不同嵌套深度和说明/排除项场景。不能只写泛泛的编码规则描述，必须给出可参照的 input→output 对照。",
    "hierarchy_hints 中的提取示例，code 字段必须展示完整的层级拼接路径（如 3A101.c.1 而非 c.1 或 1），这是大模型构图时最重要的参照。",
]
