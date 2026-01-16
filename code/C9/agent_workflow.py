import os
import operator
import json
from typing import TypedDict, List, Annotated, Optional, Dict, Union

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 导入底层 RAG 系统
# ==========================================
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import AdvancedGraphRAGSystem

# ==========================================
# 2. 初始化全局实例 (RAG & LLM)
# ==========================================

# 初始化 RAG 系统 (作为 Chef 的核心工具)
print("正在初始化 RAG 系统...")
rag_system = AdvancedGraphRAGSystem()

# 初始化控制流程用的 LLM (用于营养师审核和采购员分析)
llm = ChatOpenAI(
    model="deepseek-chat",  # 或 gpt-4
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
)

# ==========================================
# 3. 定义图的状态 (State)
# ==========================================
class AgentState(TypedDict):
    request: str                          # 用户原始需求
    history: Annotated[List[BaseMessage], operator.add] # 消息历史
    recipe_content: Optional[str]         # Chef 生成的食谱内容
    critique_feedback: Optional[str]      # 营养师的反馈意见
    is_approved: bool                     # 是否通过营养师审核
    inventory: Dict[str, str]             # 冰箱库存 (模拟数据)
    shopping_list: Optional[str]          # 最终购物清单
    iteration_count: int                  # 循环次数 (防止死循环)

# ==========================================
# 4. 定义节点 (Agent Nodes)
# ==========================================

def chef_node(state: AgentState):
    """
    👨‍🍳 主厨 Agent: 调用 RAG 系统生成或修改食谱
    """
    print("\n--- 👨‍🍳 Chef (主厨) 正在思考 ---")
    query = state["request"]
    feedback = state.get("critique_feedback")
    iteration = state.get("iteration_count", 0)

    # 如果有反馈，说明是修改阶段
    if feedback:
        print(f"👨‍🍳 Chef: 收到营养师反馈 '{feedback}'，正在调整食谱...")
        # 构造一个包含反馈的新查询，引导 RAG 系统重新检索或生成
        refined_query = f"用户原需求：{query}。修改意见：{feedback}。请重新推荐一道符合要求的菜谱，并附带详细做法。"
        # 调用 RAG 系统的问答接口
        result, _ = rag_system.ask_question_with_routing(refined_query)
    else:
        print(f"👨‍🍳 Chef: 收到用户要求 '{query}'，正在检索图谱...")
        result, _ = rag_system.ask_question_with_routing(query)

    return {
        "recipe_content": result,
        "iteration_count": iteration + 1,
        "critique_feedback": None  # 清除已处理的反馈
    }

def nutritionist_node(state: AgentState):
    """
    👩‍⚕️ 营养师 Agent: 审核食谱健康指标
    """
    print("\n--- 👩‍⚕️ Nutritionist (营养师) 正在审核 ---")
    recipe = state["recipe_content"]
    user_request = state["request"]

    # Prompt 设计：让 LLM 扮演挑剔的营养师
    system_prompt = """你是一名严格的营养师。你的任务是审核主厨的食谱是否符合用户的健康目标。
    - 如果食谱符合用户需求（如减脂、低糖、增肌等），请只回复 "APPROVE"。
    - 如果不符合，请给出简短、具体的修改建议（例如："脂肪含量过高，建议将炸鸡改为煎鸡胸肉"）。
    """
    
    user_message = f"用户需求：{user_request}\n\n主厨食谱：\n{recipe}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])
    
    content = response.content.strip()

    if "APPROVE" in content.upper():
        print("👩‍⚕️ Nutritionist: 审核通过 ✅")
        return {"is_approved": True, "critique_feedback": None}
    else:
        print(f"👩‍⚕️ Nutritionist: 审核不通过 ❌。建议：{content}")
        return {"is_approved": False, "critique_feedback": content}

def shopper_node(state: AgentState):
    """
    🛒 采购员 Agent: 对比库存生成清单
    """
    print("\n--- 🛒 Shopper (采购员) 正在盘点 ---")
    recipe = state["recipe_content"]
    inventory = state.get("inventory", {})

    prompt = f"""
    你是一名精明的家庭采购员。
    
    当前食谱内容：
    {recipe}
    
    家里的冰箱库存：
    {json.dumps(inventory, ensure_ascii=False)}
    
    任务：
    1. 提取食谱中需要的所有食材。
    2. 对比库存，忽略已有且充足的食材。
    3. 生成一份【极简补货清单】，只包含需要购买的物品和数量。
    
    请直接输出清单内容，不要废话。
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    shopping_list = response.content
    
    print(f"🛒 Shopper: 清单已生成:\n{shopping_list}")
    return {"shopping_list": shopping_list}

def sms_node(state: AgentState):
    """
    📱 SMS Agent: 发送短信 (集成 Twilio)
    """
    print("\n--- 📱 SMS Agent 正在发送 ---")
    content = state["shopping_list"]
    
    # 尝试导入 Twilio，如果未安装则模拟发送
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_FROM_NUMBER')
        to_number = os.getenv('USER_PHONE_NUMBER')
        
        if all([account_sid, auth_token, from_number, to_number]):
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body=f"【智能美食助手】您的购物清单：\n{content}",
                from_=from_number,
                to=to_number
            )
            print(f"📱 SMS: 发送成功! SID: {message.sid}")
        else:
            print("📱 SMS: 未配置 Twilio 环境变量，模拟发送成功。")
            
    except ImportError:
        print("📱 SMS: 未安装 twilio 库 (pip install twilio)，模拟发送成功。")
    except Exception as e:
        print(f"📱 SMS: 发送失败: {e}")
        
    return {}

# ==========================================
# 5. 构建图 (Graph Construction)
# ==========================================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("chef", chef_node)
workflow.add_node("nutritionist", nutritionist_node)
workflow.add_node("shopper", shopper_node)
workflow.add_node("sms", sms_node)

# 设置入口点
workflow.set_entry_point("chef")


workflow.add_edge("chef", END)


workflow.add_edge("chef", "nutritionist")

# 条件边：营养师审核逻辑
def route_after_critique(state: AgentState):
    # 如果审核通过，进入人工确认阶段（即暂停，等待进入 shopper）
    if state.get("is_approved"):
        return "approved"
    # 防止死循环：如果循环超过3次，强制通过
    if state["iteration_count"] > 3:
        print("--- ⚠️ 达到最大重试次数，强制进入下一步 ---")
        return "approved"
    # 否则回退给主厨重做
    return "rejected"

workflow.add_conditional_edges(
    "nutritionist",
    route_after_critique,
    {
        "approved": "shopper",  # 这里虽然指向 shopper，但我们会用 interrupt 拦截
        "rejected": "chef"
    }
)

workflow.add_edge("shopper", "sms")
workflow.add_edge("sms", END)

# 编译图
# 【关键点】interrupt_before=["shopper"] 实现了 Human-in-the-loop
# 系统会在进入 Shopper 节点前暂停，等待用户确认
memory = MemorySaver()
app = workflow.compile(
    interrupt_before=["shopper"],
    checkpointer=memory
    )

# ==========================================
# 6. 执行入口 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 启动前必须先初始化底层 RAG
    print("🚀 系统启动中，正在初始化知识库...")
    rag_system.initialize_system()
    rag_system.build_knowledge_base()
    
    # 2. 准备初始输入
    user_input = input("\n您想吃什么: ")
    
    # 模拟冰箱库存
    mock_inventory = {"鸡蛋": "5个", "盐": "充足", "酱油": "充足", "生菜": "1颗"}
    
    initial_state = AgentState(
        request=user_input,
        history=[],
        recipe_content=None,
        critique_feedback=None,
        is_approved=False,
        inventory=mock_inventory,
        shopping_list=None,
        iteration_count=0
    )
    
    # 配置线程 ID (用于 LangGraph 记忆)
    thread_config = {"configurable": {"thread_id": "session_1"}}
    
    print("\n--- 🔄 开始第一阶段：生成与审核 ---")
    
    # 3. 运行第一阶段 (直到遇到 interrupt 暂停)
    for event in app.stream(initial_state, thread_config):
        # event 是一个字典，key 是节点名 (如 'chef'), value 是该节点的返回结果
        for node_name, node_output in event.items():
            if node_name == "chef":
                print(f"\n🥘 主厨生成的食谱:\n{node_output.get('recipe_content')}")
            
    #         if node_name == "nutritionist":
    #             is_approved = node_output.get('is_approved')
    #             feedback = node_output.get('critique_feedback')
    #             status = "✅ 通过" if is_approved else f"❌ 驳回 (意见: {feedback})"
    #             print(f"\n📋 营养师审核结果: {status}")
    
    # # 4. Human-in-the-loop: 人工确认
    # # 获取当前暂停时的状态快照
    # snapshot = app.get_state(thread_config)
    
    # if snapshot.values.get("recipe_content"):
    #     print("\n" + "="*50)
    #     print("📝 【最终确认】营养师审核通过的食谱：")
    #     print(snapshot.values["recipe_content"])
    #     print("="*50)
        
    #     confirm = input("\n👤 人工确认: 是否满意该食谱并生成购物清单发送短信? (y/n): ")
        
    #     if confirm.lower() == "y":
    #         print("\n✅ 用户确认。继续执行：生成清单 -> 发送短信...")
    #         # 继续执行剩余步骤 (Shopper -> SMS)
    #         for event in app.stream(None, thread_config):
    #             pass
    #         print("\n🎉 流程结束！")
    #     else:
    #         print("\n🛑 用户取消，流程结束。")
    # else:
    #     print("\n⚠️ 流程异常结束，未生成食谱。")

    # 退出时清理资源
    rag_system._cleanup()