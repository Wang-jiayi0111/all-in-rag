import sys
# 确保可以导入 agent_workflow 和 main
sys.path.append(".") 

from agent_workflow import app, rag_system, AgentState

def main():
    # 1. 初始化 RAG 系统 (确保数据库连接和索引加载)
    print("正在初始化底层 RAG 系统...")
    rag_system.initialize_system()
    rag_system.build_knowledge_base()
    
    # 2. 准备初始状态
    user_input = input("\n请输入您的美食需求 (例如: 我想吃减脂餐，最好有鸡肉): ")
    
    initial_state = AgentState(
        request=user_input,
        history=[],
        iteration_count=0,
        inventory={"鸡蛋": "5个", "鸡胸肉": "0", "青椒": "2个", "盐": "充足"}, # Mock 库存
        critique_feedback=None,
        is_approved_by_nutritionist=False
    )

    # 3. 启动图执行 (第一阶段：Chef -> Nutritionist Loop)
    print("\n🚀 启动 AI 主厨与营养师协作...")
    
    # 这里的 config 用于管理会话内存，这里简单处理
    thread = {"configurable": {"thread_id": "1"}}
    
    for event in app.stream(initial_state, thread):
        # 实时打印流事件（可选）
        pass

    # 4. Human-in-the-loop: 检查当前状态
    snapshot = app.get_state(thread)
    current_recipe = snapshot.values.get("recipe_content")
    
    print("\n" + "="*50)
    print("📝 最终推荐食谱：")
    print(current_recipe)
    print("="*50)
    
    # 5. 用户确认
    user_approval = input("\n您满意这份食谱并希望生成购物清单发送到手机吗？(y/n/提出修改意见): ").strip()
    
    if user_approval.lower() == 'y':
        print("\n✅ 用户确认。正在转交采购员(Shopper)...")
        # 继续执行图 (进入 Shopper -> SMS)
        # 这里的 None 表示继续执行，没有新的输入注入，但状态会延续
        for event in app.stream(None, thread):
             pass
        print("\n🎉 流程结束！")
        
    elif user_approval.lower() == 'n':
        print("\n🚫 流程已取消。")
        
    else:
        print(f"\n🔄 用户提出修改意见: {user_approval}")
        # 如果用户有意见，我们需要更新状态并回退给 Chef
        # LangGraph 允许 update_state
        app.update_state(thread, {"critique_feedback": user_approval, "iteration_count": 0})
        
        # 重新运行 (此时会重新进入 Chef)
        # 注意：这里可能需要稍微调整图结构支持从 Shopper 前跳回 Chef，
        # 或者简单的重新创建一个 stream 运行 chef 节点。
        # 最简单的做法是在代码逻辑里处理：
        print("正在根据您的意见重新生成...")
        # (这里简化处理，实际生产环境图结构应该允许 HumanFeedback -> Chef 的边)
        # 我们可以通过修改状态后，再次调用 app.stream(..., as_node="chef") 来实现

if __name__ == "__main__":
    main()